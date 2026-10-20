//! R6's single-frame denoise by method A on the GPU: block matching and 3D DCT hard thresholding
//! (`docs/algorithms/05-single-frame-denoise.md`, section A and *Yellowmoon's reading*).
//!
//! The WGSL port (`shaders/bm_dct*.wgsl`) of [`FrameDenoise::apply`] for [`DenoiseMethod::BmDct`]: the working
//! encoding `srgb(2^gain x)` of [`denoise_space`](crate::cpu::denoise_space), BT.709 Y/Cb/Cr, method A
//! ([`crate::cpu::bm_dct::denoise`]: the passes of A.5, the group, the 3D transform, the hard threshold, the
//! aggregation), back to RGB, the blend with the encoded input, the decoding. [`encode_frame_denoise`] is the
//! device-in, device-out entry point (`docs/17-gpu.md` section 3) the device chain records as R6's step
//! (`chain.rs`, `PostStep::FrameDenoise`); [`frame_denoise`] and [`frame_denoise_core`] are the host wrappers the
//! parity tests use.
//!
//! # Dispatches
//!
//! `bm_dct_prep` over the buffer (the encoding, the three Y/Cb/Cr planes, the four accumulators zeroed), one
//! `bm_dct` dispatch per pass of A.5 in the CPU's order with one workgroup of 256 threads per reference of the
//! pass, and `bm_dct_output` over the core: `2 + passes` ([`dispatches`]). The schedule is the CPU's own
//! (`cpu::bm_dct::device_plan`, the very `Ctx::passes` the reference walks): the grid anchored to the frame, the
//! extra column, row and corner, and only the references whose read span lies inside the buffer (reading A-d).
//! With `R = 1` (`s = 6`) that is 16 regular passes and up to 9 extra ones, so 18 to 27 dispatches per tile. A
//! frame under 8 px or strengths that filter no channel take one dispatch, `bm_dct_output` alone, which encodes
//! and decodes the input as the CPU does.
//!
//! # Exactness and order
//!
//! Every pixel's sums are added in the CPU's order: pass by pass (dispatches in order), at most one reference per
//! pixel and pass (references of a pass are 24 px apart), slot by slot inside a reference (a storage barrier
//! between the slots). The references and their reads are positions of the frame, so a tile with the 23 px halo
//! ([`HALO_PX`]) gives the whole frame's bits on the device (D5); a pixel no reference reached (only in the margin
//! the caller drops) keeps its input value, as on the CPU.
//!
//! # Precision against the CPU
//!
//! The CPU works in `f64`, the device in `f32` (`docs/17-gpu.md` section 2). Away from a decision the output is
//! within a few `f32` ulp of the CPU's. The method's decisions, `|c| < tau` on 512 coefficients per channel and
//! reference, the order of the 24 candidate distances and `D > theta`, are taken on values that differ by those
//! ulp, so a coefficient that lies within rounding of `tau` (or two candidates within rounding of each other) can
//! go the other way; such a flip moves the reference's 8 blocks by up to about `tau / 512` in the encoding. What
//! sets the rate is the input of the transform: the CPU's encoded samples are `f32` too, but its Y/Cb/Cr are
//! `f64`, the device's `f32`. Measured with the thresholds near the dense middle of the coefficients (`tau`
//! about 1.3 noise deviations, 2.4 million samples), the rate did not move when the arithmetic was made more
//! exact (each group shifted by its corner value before the transform, which moves the DC term alone in exact
//! arithmetic and keeps the transform's values small) or less exact (the device's `pow` in the encoding): 4952,
//! 4949 and 5177 samples over 1e-5. So the kernel is the plain port, and its own `pow` is there for Metal
//! (below). The numbers at the render's thresholds are in `tests/gpu_bm_dct.rs`.
//!
//! # Fast-math (Metal)
//!
//! Every product, sum and difference that feeds a decision or the output is formed behind the OR barrier
//! (`bm_op`, `docs/17-gpu.md` K8), every division is correctly rounded (`bm_div`), and the power, `log2` and `exp2`
//! of the encoding are basic operations behind the barrier (`bm_pow`), so the arithmetic is one IEEE `f32`
//! sequence on every backend and Metal takes the decisions lavapipe takes. The comparisons themselves (`abs(v) <
//! tau`, `bd[q] > di`, `bd[q] <= theta`, `w > 0`) compare pinned values and are not changed by fast-math; NaN
//! and infinity are tested on the bits. In a clang model of naga's MSL of the shared functions (`docs/17-gpu.md`
//! section 13), fast-math builds with contraction and approximate reciprocals give the strict bits in every one
//! of 58 million outputs, and 20 to 35 % differ without the barriers.
//!
//! # Memory
//!
//! For a buffer of `n` pixels: the input's 3 planes, 3 Y/Cb/Cr and 4 accumulator planes of scratch (`f32`, dense)
//! and the core's 3 output planes, at most 13 `f32` per pixel ([`BYTES_PER_PIXEL`]); the largest binding is the
//! accumulators', 16 bytes per pixel ([`max_binding_bytes`]).

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::bm_dct::{self, BmDctParams, DevicePlan};
use crate::cpu::denoise_space::{WORKING_GAIN_EV, knee_and_slope};
use crate::cpu::frame_denoise::{DenoiseMethod, FrameDenoise};
use crate::cpu::tiling::Rect;
use crate::cpu::ycbcr::{CB_DIV, CR_DIV, KB, KG, KR};
use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The halo of method A in pixels (`cpu::bm_dct::HALO_PX`): a tile needs this much context on every side
/// that is not a frame edge for its core to equal the whole frame's.
pub const HALO_PX: usize = bm_dct::HALO_PX;

/// Device bytes one buffer pixel needs at most: 3 input planes, 3 Y/Cb/Cr and 4 accumulator planes, 3
/// output planes, all `f32`.
pub const BYTES_PER_PIXEL: u64 = 4 * (3 + 3 + 4 + 3);

/// Threads per workgroup of the pass kernel (one workgroup per reference block).
const PASS_THREADS: u32 = 256;

/// The power-segment constants of the sRGB curve of `denoise_space::srgb_encode` (`1.055 v^(1/2.4) - 0.055`).
/// A unit test holds the curve made from them to `srgb_encode` and `srgb_decode` bit for bit.
const SRGB_A: f64 = 1.055;
/// See [`SRGB_A`].
const SRGB_B: f64 = 0.055;
/// See [`SRGB_A`].
const SRGB_GAMMA: f64 = 2.4;

/// The uniform block of `bm_dct.wgsl`, shared by the three kernels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BmU {
    /// Buffer width, height, frame `x0`, `y0`.
    buf: [u32; 4],
    /// Frame width, height, scratch plane length (`width * height`), the barrier word (0).
    frame: [u32; 4],
    /// First reference column and row of the pass (frame coordinates), their spacings.
    refs: [u32; 4],
    /// Switches (1 luma, 2 chroma, 4 blend), mode (output: 0 encode and decode, 1 filtered).
    ctl: [u32; 4],
    /// Core `x`, `y`, width and height in the buffer.
    core: [u32; 4],
    /// Input row stride, input plane length, output row stride, output plane length.
    io: [u32; 4],
    /// `tau Y`, `tau C`, `theta`, the blend weight `B / 100`.
    tau: [f32; 4],
    /// `K(i, j)`, row-major.
    kwin: [[f32; 4]; 16],
}

/// The method A parameters of a stage.
fn params_of(fd: &FrameDenoise) -> BmDctParams {
    BmDctParams {
        luma: fd.luma,
        chroma: fd.chroma,
        radius: fd.radius,
        blend: fd.blend,
    }
}

/// Checks the stage.
fn check(fd: &FrameDenoise) -> Result<(), GpuError> {
    if fd.method != DenoiseMethod::BmDct {
        return Err(GpuError::InvalidInput(format!(
            "the device runs the single-frame denoise of method A only, the stage asks for {}",
            fd.method.name()
        )));
    }
    if !(fd.luma.is_finite() && fd.chroma.is_finite() && fd.blend.is_finite()) {
        return Err(GpuError::InvalidInput(format!(
            "method A: the strengths must be finite, got {fd:?}"
        )));
    }
    Ok(())
}

/// The plan of a buffer, `None` when the method changes nothing there.
fn plan_of(width: usize, height: usize, place: Placement, fd: &FrameDenoise) -> Option<DevicePlan> {
    bm_dct::device_plan(width, height, place, &params_of(fd))
}

/// The dispatches [`encode_frame_denoise`] records for a `width` x `height` buffer at `place`: the prep, one
/// per pass and the output, or the output alone when the method changes nothing.
#[must_use]
pub fn dispatches(width: usize, height: usize, place: Placement, fd: &FrameDenoise) -> usize {
    plan_of(width, height, place, fd).map_or(1, |p| 2 + p.passes.len())
}

/// The largest storage binding (bytes) the stage needs for a buffer of `rect` pixels: the four accumulator
/// planes (the input's three planes, the Y/Cb/Cr scratch and the output are smaller). A test checks it against
/// the 128 MiB design binding at [`crate::chain::MAX_PHYS_AREA`] (`docs/17-gpu.md` K4).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    4 * (rect.w * rect.h) as u64 * 4
}

/// The first value, the count and the spacing of the references of one axis of a pass (an arithmetic
/// progression: a run of the grid, or one extra reference).
fn progression(v: &[usize]) -> Result<(u32, u32, u32), GpuError> {
    let (Some(&first), Some(&last)) = (v.first(), v.last()) else {
        return Err(GpuError::InvalidInput(
            "method A: a pass without references".into(),
        ));
    };
    let step = if v.len() > 1 { v[1] - v[0] } else { 0 };
    if v.windows(2).any(|w| w[1] - w[0] != step) || first + step * (v.len() - 1) != last {
        return Err(GpuError::InvalidInput(format!(
            "method A: the references {v:?} of a pass are not evenly spaced"
        )));
    }
    Ok((first as u32, v.len() as u32, step as u32))
}

/// The shapes of the input and the output of one stage, in the units of the uniform block.
#[derive(Clone, Copy, Debug)]
struct Shapes {
    /// Buffer width and height.
    size: (usize, usize),
    /// The buffer's place in the frame.
    place: Placement,
    /// Input row stride and plane length.
    input: (usize, usize),
    /// Output row stride and plane length.
    output: (usize, usize),
}

impl Shapes {
    fn of(src: &GpuImage, out: &GpuImage) -> Shapes {
        Shapes {
            size: (src.width, src.height),
            place: src.place,
            input: (src.stride, src.plane_len()),
            output: (out.stride, out.plane_len()),
        }
    }
}

/// The uniform block common to the three kernels.
fn base_uniform(sh: Shapes, core: Rect, plan: Option<&DevicePlan>, fd: &FrameDenoise) -> BmU {
    let (window, ..) = bm_dct::device_constants();
    let mut kwin = [[0.0f32; 4]; 16];
    for (q, k) in window.iter().flatten().enumerate() {
        kwin[q / 4][q % 4] = *k as f32;
    }
    let place = sh.place;
    let (width, height) = sh.size;
    let (switches, tau) = match plan {
        Some(p) => {
            let blend = fd.blend > 0.0;
            let switches = u32::from(p.active[0])
                | (u32::from(p.active[1] && p.active[2]) << 1)
                | (u32::from(blend) << 2);
            let t = if blend {
                (fd.blend / 100.0).min(1.0) as f32
            } else {
                0.0
            };
            (
                switches,
                [p.tau[0] as f32, p.tau[1] as f32, p.theta as f32, t],
            )
        }
        None => (0, [0.0; 4]),
    };
    BmU {
        buf: [
            width as u32,
            height as u32,
            place.x0 as u32,
            place.y0 as u32,
        ],
        frame: [
            place.frame_width as u32,
            place.frame_height as u32,
            (width * height) as u32,
            // The barrier word: +0.0, so that OR-ing it changes no bit.
            0,
        ],
        refs: [0; 4],
        ctl: [switches, u32::from(plan.is_some()), 0, 0],
        core: [core.x as u32, core.y as u32, core.w as u32, core.h as u32],
        io: [
            sh.input.0 as u32,
            sh.input.1 as u32,
            sh.output.0 as u32,
            sh.output.1 as u32,
        ],
        tau,
        kwin,
    }
}

/// Records R6's single-frame denoise by method A of the `core` rectangle of `src` (device in, device out) and
/// returns the core as a new three-plane image whose placement is the input's moved to the core: the device
/// form of [`FrameDenoise::apply`] followed by the crop to `core`.
///
/// `src` is the balanced linear RGB of the buffer the method reads, placed in the frame (`src.place`): the whole
/// frame, or a window of it with [`HALO_PX`] px of context around `core` on every side that is not a frame edge.
/// The method's grid, its candidates and the references it computes are positions of the frame (readings G4,
/// A-a and A-d), so the core's pixels equal those of a run over the whole frame bit for bit.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a stage that is not method A or has strengths that are not finite, an input that
/// is not three planes or does not lie inside its frame, an empty core or one outside the input, or a buffer
/// whose four accumulator planes do not fit a 32-bit index; the pool's errors; shader errors.
pub fn encode_frame_denoise(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    core: Rect,
    fd: &FrameDenoise,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    check(fd)?;
    if src.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "method A: the input has {} planes, not 3",
            src.planes
        )));
    }
    if core.is_empty() || core.x1() > src.width || core.y1() > src.height {
        return Err(GpuError::InvalidInput(format!(
            "method A: the core {core:?} is empty or outside the {} x {} input",
            src.width, src.height
        )));
    }
    let place = src.place;
    if place.x0 + src.width > place.frame_width || place.y0 + src.height > place.frame_height {
        return Err(GpuError::InvalidInput(format!(
            "method A: the {} x {} input at {place:?} does not lie inside its frame",
            src.width, src.height
        )));
    }
    let n = src.width * src.height;
    if 4 * n as u64 > u64::from(u32::MAX) {
        return Err(GpuError::InvalidInput(format!(
            "method A: {n} pixels do not fit a 32-bit index"
        )));
    }
    let out = GpuImage::alloc(
        ctx,
        "frame denoise rgb",
        core.w,
        core.h,
        3,
        place.offset(core.x, core.y),
    )?;
    let core_groups = (
        (core.w as u32).div_ceil(GROUP),
        (core.h as u32).div_ceil(GROUP),
        1,
    );
    let plan = plan_of(src.width, src.height, place, fd);
    let u = base_uniform(Shapes::of(src, &out), core, plan.as_ref(), fd);
    let Some(plan) = plan else {
        // The method changes nothing: the input encoded and decoded, as the CPU's stage gives it.
        let ub = passes.uniform("frame denoise uniform", &u)?;
        passes.dispatch(
            KernelId::BmDctOutput,
            "frame denoise output",
            &[
                (0, ub.binding()),
                (1, src.buf.binding()),
                (2, src.buf.binding()),
                (3, src.buf.binding()),
                (4, out.buf.binding()),
            ],
            core_groups,
        )?;
        return Ok(out);
    };
    let ycc = passes.scratch("frame denoise ycc", 3 * n as u64 * 4)?;
    let acc = passes.scratch("frame denoise accumulators", 4 * n as u64 * 4)?;
    let ub = passes.uniform("frame denoise uniform", &u)?;
    passes.dispatch(
        KernelId::BmDctPrep,
        "frame denoise prep",
        &[
            (0, ub.binding()),
            (1, src.buf.binding()),
            (2, ycc.binding()),
            (3, acc.binding()),
        ],
        (
            (src.width as u32).div_ceil(GROUP),
            (src.height as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    for (ys, xs) in &plan.passes {
        let (x0, nx, sx) = progression(xs)?;
        let (y0, ny, sy) = progression(ys)?;
        let mut pu = u;
        pu.refs = [x0, y0, sx, sy];
        let pb = passes.uniform("frame denoise pass uniform", &pu)?;
        passes.dispatch(
            KernelId::BmDct,
            "frame denoise pass",
            &[(0, pb.binding()), (2, ycc.binding()), (3, acc.binding())],
            (nx, ny, 1),
        )?;
    }
    passes.dispatch(
        KernelId::BmDctOutput,
        "frame denoise output",
        &[
            (0, ub.binding()),
            (1, src.buf.binding()),
            (2, ycc.binding()),
            (3, acc.binding()),
            (4, out.buf.binding()),
        ],
        core_groups,
    )?;
    Ok(out)
}

/// The stage on the host for the whole of `img`: uploads it, runs [`encode_frame_denoise`] and reads the result
/// back. The GPU counterpart of [`FrameDenoise::apply`].
///
/// # Errors
///
/// As [`encode_frame_denoise`], plus the upload's and the readback's.
pub fn frame_denoise(
    ctx: &GpuContext,
    img: &RgbImage,
    fd: &FrameDenoise,
) -> Result<RgbImage, GpuError> {
    frame_denoise_core(ctx, img, Rect::whole(img.width, img.height), fd)
}

/// [`frame_denoise`] that keeps the `core` rectangle of `img` only (`FrameDenoise::apply` and then
/// `RgbImage::extract(core)` on the CPU).
///
/// # Errors
///
/// As [`frame_denoise`].
pub fn frame_denoise_core(
    ctx: &GpuContext,
    img: &RgbImage,
    core: Rect,
    fd: &FrameDenoise,
) -> Result<RgbImage, GpuError> {
    let input = GpuImage::upload_rgb(ctx, "frame denoise input", img)?;
    let mut passes = PassList::new(ctx, "frame denoise");
    let out = encode_frame_denoise(&mut passes, &input, core, fd)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    drop(input);
    Ok(RgbImage {
        width: out.width,
        height: out.height,
        place: out.place,
        data,
    })
}

/// `x` as an `f32` WGSL literal.
fn lit(x: f64) -> String {
    format!("{:?}", x as f32)
}

/// `x` as the pair `hi + lo` of `f32` literals (`hi` the nearest `f32`, `lo` the rest rounded).
fn pair(x: f64) -> (String, String) {
    let hi = x as f32;
    (lit(f64::from(hi)), lit(x - f64::from(hi)))
}

fn vec4_lit(v: [f64; 4]) -> String {
    format!(
        "vec4<f32>({}, {}, {}, {})",
        lit(v[0]),
        lit(v[1]),
        lit(v[2]),
        lit(v[3])
    )
}

/// The constants of the kernels, written from the CPU reference (`docs/17-gpu.md` K7): those of
/// `working_encoding.wgsl`, which the false-colour gate (`kernels::false_color`) includes too, and the DCT
/// matrix.
pub(super) fn constants() -> String {
    use std::fmt::Write as _;
    let (knee_x, slope, knee_y) = knee_and_slope();
    let (_, even, odd, gain) = bm_dct::device_constants();
    let ln2 = std::f64::consts::LN_2;
    let mut s = String::new();
    for (name, v) in [
        ("BM_GAIN", WORKING_GAIN_EV.exp2()),
        ("BM_KNEE_X", knee_x),
        ("BM_SLOPE", slope),
        ("BM_KNEE_Y", knee_y),
        ("BM_SRGB_A", SRGB_A),
        ("BM_SRGB_B", SRGB_B),
        ("BM_KR", KR),
        ("BM_KG", KG),
        ("BM_KB", KB),
        ("BM_CB_DIV", CB_DIV),
        ("BM_CR_DIV", CR_DIV),
        ("BM_INV_RT_GAIN", 1.0 / gain),
        ("BM_TWO_OVER_LN2", 2.0 / ln2),
    ] {
        let _ = writeln!(s, "const {name}: f32 = {};", lit(v));
    }
    for (name, v) in [("BM_GAMMA", SRGB_GAMMA), ("BM_INV_GAMMA", 1.0 / SRGB_GAMMA)] {
        let (hi, lo) = pair(v);
        let _ = writeln!(s, "const {name}_HI: f32 = {hi};");
        let _ = writeln!(s, "const {name}_LO: f32 = {lo};");
    }
    for j in [3u32, 5, 7, 9] {
        let _ = writeln!(s, "const BM_ATANH_{j}: f32 = {};", lit(1.0 / f64::from(j)));
    }
    let mut factorial = 1.0f64;
    for k in 1..=7i32 {
        factorial *= f64::from(k);
        let _ = writeln!(
            s,
            "const BM_EXP2_{k}: f32 = {};",
            lit(ln2.powi(k) / factorial)
        );
    }
    // Rows 2m and 2m + 1 of the DCT matrix (the forward transform's dot products) and their columns (the
    // inverse's).
    for m in 0..4 {
        let _ = writeln!(s, "const BM_E{m}: vec4<f32> = {};", vec4_lit(even[m]));
        let _ = writeln!(s, "const BM_O{m}: vec4<f32> = {};", vec4_lit(odd[m]));
    }
    for n in 0..4 {
        let col = |t: &[[f64; 4]; 4]| [t[0][n], t[1][n], t[2][n], t[3][n]];
        let _ = writeln!(s, "const BM_EC{n}: vec4<f32> = {};", vec4_lit(col(&even)));
        let _ = writeln!(s, "const BM_OC{n}: vec4<f32> = {};", vec4_lit(col(&odd)));
    }
    s
}

/// The WGSL of a method A kernel: the constants, the shared file, the kernel's own file and its `main`.
pub(super) fn body(id: KernelId) -> String {
    let (section, main) = match id {
        KernelId::BmDctPrep => (
            include_str!("../shaders/bm_dct_prep.wgsl"),
            format!(
                "@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
                 fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ bm_prep(gid); }}\n"
            ),
        ),
        KernelId::BmDct => (
            include_str!("../shaders/bm_dct_pass.wgsl"),
            format!(
                "@compute @workgroup_size({PASS_THREADS}, 1, 1)\n\
                 fn main(@builtin(workgroup_id) wid: vec3<u32>, \
                 @builtin(local_invocation_index) lid: u32) {{ bm_pass(wid, lid); }}\n"
            ),
        ),
        KernelId::BmDctOutput => (
            include_str!("../shaders/bm_dct_output.wgsl"),
            format!(
                "@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
                 fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ bm_output(gid); }}\n"
            ),
        ),
        _ => return String::new(),
    };
    let mut s = constants();
    s.push_str(include_str!("../shaders/bm_dct.wgsl"));
    s.push('\n');
    s.push_str(include_str!("../shaders/working_encoding.wgsl"));
    s.push('\n');
    s.push_str(section);
    s.push('\n');
    s.push_str(&main);
    s
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::cpu::denoise_space::{srgb_decode, srgb_encode};
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    const IDS: [KernelId; 3] = [KernelId::BmDctPrep, KernelId::BmDct, KernelId::BmDctOutput];

    fn module_of(id: KernelId) -> (naga::Module, naga::valid::ModuleInfo) {
        let source = shader_source(StorageMode::F32, id);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{id:?}: {}", e.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{id:?}: {e:?}"));
        (module, info)
    }

    #[test]
    fn each_kernel_binds_exactly_the_buffers_the_host_binds() {
        let table: [(KernelId, &[u32], [u32; 3]); 3] = [
            (KernelId::BmDctPrep, &[0, 1, 2, 3], [GROUP, GROUP, 1]),
            (KernelId::BmDct, &[0, 2, 3], [PASS_THREADS, 1, 1]),
            (KernelId::BmDctOutput, &[0, 1, 2, 3, 4], [GROUP, GROUP, 1]),
        ];
        for (id, slots, size) in table {
            let (module, info) = module_of(id);
            assert_eq!(module.entry_points.len(), 1, "{id:?}");
            assert_eq!(module.entry_points[0].workgroup_size, size, "{id:?}");
            let used = info.get_entry_point(0);
            let mut bindings = BTreeSet::new();
            for (handle, var) in module.global_variables.iter() {
                if let Some(b) = &var.binding
                    && !used[handle].is_empty()
                {
                    bindings.insert(b.binding);
                }
            }
            assert_eq!(
                bindings,
                slots.iter().copied().collect::<BTreeSet<_>>(),
                "{id:?}"
            );
        }
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_bm_u() {
        for id in IDS {
            let (module, _) = module_of(id);
            let var = module
                .global_variables
                .iter()
                .map(|(_, v)| v)
                .find(|v| v.name.as_deref() == Some("bp"))
                .expect("the uniform `bp`");
            let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
                panic!("`bp` is not a struct");
            };
            let want = [
                ("buf", offset_of!(BmU, buf)),
                ("frame", offset_of!(BmU, frame)),
                ("refs", offset_of!(BmU, refs)),
                ("ctl", offset_of!(BmU, ctl)),
                ("core", offset_of!(BmU, core)),
                ("io", offset_of!(BmU, io)),
                ("tau", offset_of!(BmU, tau)),
                ("kwin", offset_of!(BmU, kwin)),
            ];
            assert_eq!(members.len(), want.len(), "{id:?}");
            for (m, (name, offset)) in members.iter().zip(want) {
                assert_eq!(m.name.as_deref(), Some(name));
                assert_eq!(m.offset as usize, offset, "{id:?}: offset of {name}");
            }
            assert_eq!(*span as usize, size_of::<BmU>(), "{id:?}");
        }
    }

    /// The sRGB constants typed here, in the formula of `denoise_space`, give its curve bit for bit.
    #[test]
    fn the_srgb_constants_are_those_of_the_working_encoding() {
        let (knee_x, slope, knee_y) = knee_and_slope();
        let enc = |v: f64| {
            if v < knee_x {
                slope * v
            } else {
                SRGB_A * v.powf(1.0 / SRGB_GAMMA) - SRGB_B
            }
        };
        let dec = |y: f64| {
            if y < knee_y {
                y / slope
            } else {
                ((y + SRGB_B) / SRGB_A).powf(SRGB_GAMMA)
            }
        };
        for i in -200..4000 {
            let v = f64::from(i) * 1.37e-3;
            assert_eq!(enc(v).to_bits(), srgb_encode(v).to_bits(), "encode {v}");
            assert_eq!(dec(v).to_bits(), srgb_decode(v).to_bits(), "decode {v}");
        }
    }

    #[test]
    fn the_shader_constants_are_the_cpus() {
        let src = shader_source(StorageMode::F32, KernelId::BmDct);
        let constant = |name: &str| -> f32 {
            let start = src
                .find(&format!("const {name}: f32 = "))
                .unwrap_or_else(|| panic!("const {name}"));
            let text = &src[start..];
            let value = &text[text.find("= ").expect("=") + 2..text.find(';').expect(";")];
            value.parse::<f32>().expect("a number")
        };
        let (knee_x, slope, knee_y) = knee_and_slope();
        assert_eq!(constant("BM_GAIN"), WORKING_GAIN_EV.exp2() as f32);
        assert_eq!(constant("BM_KNEE_X"), knee_x as f32);
        assert_eq!(constant("BM_SLOPE"), slope as f32);
        assert_eq!(constant("BM_KNEE_Y"), knee_y as f32);
        assert_eq!(constant("BM_CB_DIV"), CB_DIV as f32);
        assert_eq!(constant("BM_INV_RT_GAIN"), 1.0 / 512.0);
        let hi = constant("BM_GAMMA_HI");
        assert_eq!(hi, SRGB_GAMMA as f32);
        assert_eq!(constant("BM_GAMMA_LO"), (SRGB_GAMMA - f64::from(hi)) as f32);
        let hi = constant("BM_INV_GAMMA_HI");
        assert_eq!(hi, (1.0 / SRGB_GAMMA) as f32);
        assert_eq!(
            constant("BM_INV_GAMMA_LO"),
            (1.0 / SRGB_GAMMA - f64::from(hi)) as f32
        );
        let (_, even, odd, _) = bm_dct::device_constants();
        let line = format!("const BM_O1: vec4<f32> = {};", vec4_lit(odd[1]));
        assert!(src.contains(&line), "{line}");
        let line = format!(
            "const BM_EC2: vec4<f32> = {};",
            vec4_lit([even[0][2], even[1][2], even[2][2], even[3][2]])
        );
        assert!(src.contains(&line), "{line}");
    }

    #[test]
    fn the_decisions_and_the_sums_are_hidden_from_the_optimiser() {
        // Metal compiles with fast-math: the sums that the hard threshold, the distance order and theta
        // compare, and the quotients, must stay behind `bm_op`. Lavapipe does not re-associate, so only the
        // text can pin it.
        let common = include_str!("../shaders/bm_dct.wgsl");
        let encoding = include_str!("../shaders/working_encoding.wgsl");
        let pass = include_str!("../shaders/bm_dct_pass.wgsl");
        let output = include_str!("../shaders/bm_dct_output.wgsl");
        for (src, needle) in [
            (common, "return bitcast<f32>(bitcast<u32>(x) | bp.frame.w);"),
            (encoding, "let s = bm_div(bm_op(m - 1.0), bm_op(m + 1.0));"),
            (encoding, "return bm_op(q + bm_op(c1 + bm_op(rem2 / d)));"),
            (encoding, "let v = bm_op(x * BM_GAIN);"),
            (pass, "acc = bm_op(acc + bm_op(r.y * b));"),
            (pass, "let s0 = bm_op(x[0] + x[7]);"),
            (pass, "bm_op(e0 - o0),"),
            (
                pass,
                "p = bm_op(p + bm_op(bm_kwin(i * 8u + j) * abs(bm_op(bm_win[ro + j] - bm_win[co + j]))));",
            ),
            (pass, "d = bm_op(d + bm_part[at + i]);"),
            (pass, "if (abs(y[q]) < tau) {"),
            (pass, "bm_acc[at] = bm_op(bm_acc[at] + term);"),
            (output, "yv = bm_div(bm_acc[i], w);"),
            (common, "return bm_op(d + bm_op(bm_op(e - d) * t));"),
            (
                common,
                "let g = bm_div(bm_op(bm_op(y - bm_op(BM_KR * r)) - bm_op(BM_KB * b)), BM_KG);",
            ),
            (pass, "return bm_op(bm_op(kk * v) * BM_INV_RT_GAIN);"),
        ] {
            assert!(src.contains(needle), "the shader lost `{needle}`");
        }
        for src in [common, encoding, pass, output] {
            assert!(!src.contains("fma("), "no fused multiply-add on purpose");
        }
        // The device's pow, log2 and exp2 only in the fallbacks for arguments outside the series.
        let calls = |name: &str| {
            let pattern = format!("{name}(");
            [common, encoding, pass, output]
                .iter()
                .flat_map(|src| src.lines())
                .map(|l| l.split("//").next().unwrap_or(""))
                .map(|code| {
                    code.match_indices(&pattern)
                        .filter(|(at, _)| {
                            !code[..*at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                        })
                        .count()
                })
                .sum::<usize>()
        };
        assert_eq!(calls("pow"), 0);
        assert_eq!(calls("log2"), 1);
        assert_eq!(calls("exp2"), 1);
    }

    fn stage(luma: f64, chroma: f64, radius: usize, blend: f64) -> FrameDenoise {
        FrameDenoise {
            method: DenoiseMethod::BmDct,
            luma,
            chroma,
            radius,
            blend,
        }
    }

    #[test]
    fn the_barrier_word_is_plus_zero_and_the_switches_follow_the_plan() {
        // Every pinned value is ORed with `frame.w`: any other bits would corrupt it, and -0.0 would set
        // every sign bit. Nothing but this test reads the word back.
        let place = Placement {
            x0: 30,
            y0: 20,
            frame_width: 300,
            frame_height: 200,
        };
        let sh = Shapes {
            size: (64, 48),
            place,
            input: (64, 64 * 48),
            output: (40, 40 * 30),
        };
        let core = Rect::new(12, 9, 40, 30);
        for (fd, switches, mode) in [
            (stage(30.0, 40.0, 1, 15.0), 7, 1),
            (stage(30.0, 0.0, 0, 0.0), 1, 1),
            (stage(0.0, 40.0, 3, 0.0), 2, 1),
            (stage(0.0, 0.0, 1, 15.0), 0, 0),
        ] {
            let plan = plan_of(64, 48, place, &fd);
            let u = base_uniform(sh, core, plan.as_ref(), &fd);
            assert_eq!(u.frame[3], 0.0f32.to_bits(), "{fd:?}");
            assert_eq!((u.ctl[0], u.ctl[1]), (switches, mode), "{fd:?}");
            assert_eq!(u.buf, [64, 48, 30, 20]);
            assert_eq!(u.frame[..3], [300, 200, 64 * 48]);
            if let Some(p) = plan {
                assert_eq!(u.tau[0], p.tau[0] as f32);
                assert_eq!(u.tau[2], p.theta as f32);
            }
        }
        let fd = stage(30.0, 40.0, 1, 15.0);
        let u = base_uniform(sh, core, plan_of(64, 48, place, &fd).as_ref(), &fd);
        assert_eq!(u.tau[3], 0.15);
        let (window, ..) = bm_dct::device_constants();
        assert_eq!(u.kwin[2][1], window[1][1] as f32);
        assert_eq!(u.kwin[15][3], window[7][7] as f32);
    }

    #[test]
    fn the_dispatches_are_the_prep_one_per_pass_and_the_output() {
        let whole = Placement::whole(100, 70);
        // R = 1 (s = 6): 4 x 4 regular passes; 100 - 8 = 92 is not a multiple of 6, 70 - 8 = 62 neither, so
        // the extra column (4 passes), the extra row (4) and the corner (1).
        assert_eq!(
            dispatches(100, 70, whole, &stage(30.0, 40.0, 1, 0.0)),
            2 + 16 + 9
        );
        // R = 0 (s = 8): 92 and 62 are not multiples of 8 either: 9 + 3 + 3 + 1.
        assert_eq!(
            dispatches(100, 70, whole, &stage(30.0, 40.0, 0, 0.0)),
            2 + 16
        );
        // Nothing to filter, or a frame under 8 px: the output pass alone.
        assert_eq!(dispatches(100, 70, whole, &stage(0.0, 0.0, 1, 0.0)), 1);
        assert_eq!(
            dispatches(7, 70, Placement::whole(7, 70), &stage(30.0, 40.0, 1, 0.0)),
            1
        );
    }

    #[test]
    fn evenly_spaced_references_are_a_progression_and_others_are_refused() {
        assert_eq!(progression(&[5]).expect("one"), (5, 1, 0));
        assert_eq!(progression(&[6, 30, 54]).expect("three"), (6, 3, 24));
        assert!(progression(&[6, 30, 55]).is_err());
        assert!(progression(&[]).is_err());
    }

    #[test]
    fn the_largest_binding_is_the_accumulators() {
        let r = Rect::new(0, 0, 1000, 700);
        assert_eq!(max_binding_bytes(r), 16 * 700_000);
        assert!(max_binding_bytes(r) >= GpuImage::buffer_bytes(1000, 700, 3));
    }
}

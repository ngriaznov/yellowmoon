//! The log-guided fast demosaic on the GPU (RA-09): `shaders/log_guided.wgsl`, the all-float form of
//! `docs/algorithms/02-log-guided-demosaic.md` section 3.9, against its CPU reference
//! [`cpu::demosaic::log_guided::demosaic_float`](crate::cpu::demosaic::log_guided::demosaic_float).
//!
//! # What runs where
//!
//! One shader, one pipeline ([`KernelId::LogGuided`]), four dispatches in the order of the spec's
//! section 5 (all per pixel over a small neighbourhood, no reductions, no barriers):
//!
//! | Pass | Reads | Writes |
//! |---|---|---|
//! | 0 | the mosaic, `log2` taken on the fly at +-2 | `Lh`, the horizontal guide filter |
//! | 1 | `Lh` | `C = 2^max(Lv, 0)`, the guide |
//! | 2 | `C` | `SH`, `SV`: horizontal 5-sums of the structure measures |
//! | 3 | the mosaic, `C`, `SH`, `SV` | the three output planes |
//!
//! `Lh`, `C`, `SH` and `SV` are four `f32` planes of one scratch buffer (16 bytes per pixel of the
//! rectangle). They are `f32` in **every** storage mode: the guide is a logarithm, and an `f16`
//! plane of it would have steps of 0.008 to 0.016 stop. Only the mosaic (input) and the three output
//! planes use the device's storage type (`f32`, packed `f16` or native `f16`), as the other raw
//! kernels do, so that the kernel can later sit between the front end and the colour chain without
//! a conversion. With `f16` storage the input and output are rounded to `f16`; the parity tests use
//! `f32` storage for the tight bounds and an `f16`-aware reference for the others.
//!
//! # Memory
//!
//! For a rectangle of `n` pixels: 16 bytes per pixel of scratch (`f32`), the mosaic (2 or 4 bytes
//! per pixel, by storage mode) and the three output planes (6 or 12). The scratch is one storage
//! binding, so it must fit the device's `max_storage_buffer_binding_size` (128 MiB by default: 8
//! million pixels, a tile of 2048 px with its halo needs 68 MB). Use [`demosaic_tiled`] for frames.
//!
//! # Differences from the CPU reference
//!
//! The same function in `f32` instead of `f64`. Where it is not bit-exact, and why:
//!
//! * `log2` and `exp2` are the GPU's `f32` instructions, not the platform's `f64` libm rounded to
//!   `f32`. On lavapipe their error is a few `f32` ulp.
//! * The guide filters, the box sums and the reconstruction accumulate in `f32` (the CPU: `f64`).
//!   Two exact rewrites keep the rounding error relative to the *differences* of neighbouring
//!   values instead of to the values: the filters are evaluated about the centre value
//!   (`Lh = Lc + sum t_i (L_i - Lc)`, the taps sum to exactly 1) and the structure measures as
//!   `|((C_a - C) + (C_b - C)) / 2| + p`. Without them, a guide that is exactly flat or symmetric
//!   (a one-pixel-wide strip) gave structure weights that differed by rounding noise, and a green
//!   sample that differed from the CPU by 1e-3.
//! * The green blend is the spec's `f32` form (3.9, the cube-weighted blend divided through by a
//!   weight), divided through by the *larger* of the two weights `max(D, 1)^3` and `max(E, 1)^3`, so
//!   that the other weight is `(smaller / larger)^3 <= 1`. When `D <= E` this is exactly the spec's
//!   `t = (D / E)^3`, `C (HA + t VA) / (HC + t VC)`; when `D > E` it is the same blend divided by
//!   the other weight. The spec's single form made `t` reach 1e15 on a strong horizontal edge, and
//!   `t * VC` overflowed `f32` (NaN output) for mosaic values above about 1e5, where the CPU's `f64`
//!   is finite. The CPU multiplies by the two cubed weights and divides; the forms agree up to
//!   rounding.
//! * The box mean is `sum / 25` (the float form has no floor, so the `floor(sum / 25)` of spec
//!   section 8 does not arise).
//! * Borders: every read of the mosaic or of a plane is mirrored at the rectangle's edge instead of
//!   padding the mosaic once; the spec (3.0) shows the two are the same function. Tiles that carry
//!   [`HALO_PX`] pixels of context and the whole frame give the same pixels, bit for bit (tested).
//! * `f32` range: the kernel stays finite up to mosaic values of about `3e32` and starts to overflow
//!   near `1e33`, where the 25-sample box sum of the structure measures exceeds the `f32` range (the
//!   CPU's `f32` planes: between `1e33` and `3e33`, at `A = 65535 v`). Balanced mosaics are a few
//!   times white at most.
//! * **Fast-math (Metal).** Metal's compiler re-associates by default, which turns the reconstruction
//!   `x * (a / b)` into `(x * a) / b`: with `x` and `a` both on the working scale of a very bright
//!   frame (A above 1.8e19) the product overflows although each factor and the result are finite
//!   (the Apple runners returned infinity from a mosaic scale of 1e20; 1e10 was fine). Each of the
//!   four quotients of the reconstruction (`C / HC`, `C / VC`, `C / (blend of HC, VC)`, `C / SC`) is
//!   therefore a barrier (`lg_opaque`: an xor with the always-zero `ctl.y`), so that it stays one
//!   rounded ratio near 1. Compiled with clang `-O2 -ffast-math`, these expressions gave infinity
//!   from a scale of about 3e21 on without the barrier, and the numbers of IEEE arithmetic with it.
//!   The exact rewrites of the guide filters above are not protected: re-association undoes them
//!   (a few ulp of `C` in the smooth parts), which the Apple runners passed within the bounds.
//!
//! Not wired into a render path (the log-guided demosaic is a test-switch method on the CPU only,
//! `docs/gpu-plan.md` section 5). [`encode_demosaic`] is the device-in, device-out entry point of the
//! plan's shape (`f32` planes, a [`PassList`](crate::stream::PassList)); the host wrappers upload, run and
//! read back, for the tests, and still support the `f16` storage modes of the legacy kernels.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{
    GROUP, STORAGE_IN, STORAGE_OUT, even, groups, image_bytes, image_samples, sample_bytes,
};
use crate::cpu::demosaic::log_guided::{HALO_PX, derive};
use crate::cpu::noise::WORKING_SCALE;
use crate::cpu::tiling::{Halo, Stage};
use crate::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};
use crate::device::{GpuContext, GpuError, StorageMode};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::stream::{PassList, upload_bytes};
use crate::tiles::TilePlan;

/// The four `f32` planes of the scratch buffer: `Lh`, `C`, `SH`, `SV`.
const SCRATCH_PLANES: usize = 4;

/// The taps of the guide kernel `(k, 256, 512 - 2k, 256, k)` sum to 1024 (spec 2.2); the shader
/// gets them divided by this.
const KERNEL_SUM: f32 = 1024.0;

/// The uniform block of the shader (`struct U` in `log_guided.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LogGuidedU {
    /// Width, height, row stride (even), plane length in samples.
    dims: [u32; 4],
    /// Colour (0 R, 1 G, 2 B) at local position `2 * (y & 1) + (x & 1)`.
    cfa: [u32; 4],
    /// Outer, inner and centre tap of the guide kernel, divided by 1024.
    taps: [f32; 4],
    /// The offset `p` and the working scale `S`.
    misc: [f32; 4],
    /// The pass, then a word that is always 0: the runtime zero of the shader's fast-math barrier
    /// (`lg_opaque`), which the compiler must not be able to fold.
    ctl: [u32; 4],
}

/// Checks that a mosaic can be handed to the kernel.
fn check_mosaic(m: &Mosaic) -> Result<(), GpuError> {
    if m.width == 0 || m.height == 0 || m.data.len() != m.width * m.height {
        return Err(GpuError::InvalidInput(format!(
            "log-guided demosaic: a {}x{} mosaic with {} samples",
            m.width,
            m.height,
            m.data.len()
        )));
    }
    Ok(())
}

/// Uploads `m` as one plane of samples with an even row stride, in the device's storage mode.
fn upload_mosaic(ctx: &GpuContext, m: &Mosaic) -> Result<PooledBuffer, GpuError> {
    let mode = ctx.storage_mode();
    let stride = even(m.width);
    let plane = stride * m.height;
    let mut samples = vec![0.0f32; plane];
    for (dst, src) in samples
        .chunks_exact_mut(stride)
        .zip(m.data.chunks_exact(m.width))
    {
        dst[..m.width].copy_from_slice(src);
    }
    let bytes = image_bytes(mode, &samples);
    let buf = ctx
        .pool()
        .acquire("log-guided mosaic", sample_bytes(mode, plane), STORAGE_IN)?;
    upload_bytes(ctx, &buf, 0, &bytes)?;
    Ok(buf)
}

/// Records the four dispatches over the `w x h` mosaic in `mosaic` (samples in the device's storage
/// mode, even row stride) into `passes`; the three planes are written to `rgb` in the same layout.
fn record(
    passes: &mut PassList<'_>,
    mosaic: &PooledBuffer,
    rgb: &PooledBuffer,
    (w, h): (usize, usize),
    cfa: Cfa,
    noise: NoiseDn,
    guide_detail: i32,
) -> Result<(), GpuError> {
    let stride = even(w);
    let plane = stride * h;
    // The float form: no headroom, p from the read noise at H = 1 (spec 2.2).
    let d = derive(noise, guide_detail, 0);
    let tap = |i: usize| d.kernel[i] as f32 / KERNEL_SUM;
    let base = LogGuidedU {
        dims: [w as u32, h as u32, stride as u32, plane as u32],
        cfa: std::array::from_fn(|p| cfa.color_at(p & 1, p >> 1).plane() as u32),
        taps: [tap(0), tap(1), tap(2), 0.0],
        misc: [d.p as f32, WORKING_SCALE as f32, 0.0, 0.0],
        ctl: [0; 4],
    };
    let scratch = passes.scratch(
        "log-guided scratch",
        SCRATCH_PLANES as u64 * plane as u64 * 4,
    )?;
    let per_pixel = (
        w.div_ceil(GROUP as usize) as u32,
        h.div_ceil(GROUP as usize) as u32,
    );
    let per_pair = groups(stride, h);
    let labels = [
        "log-guided Lh",
        "log-guided C",
        "log-guided SH SV",
        "log-guided rgb",
    ];
    for (pass, dispatch) in [per_pixel, per_pixel, per_pixel, per_pair]
        .into_iter()
        .enumerate()
    {
        let mut u = base;
        u.ctl[0] = pass as u32;
        let ub = passes.uniform("log-guided uniform", &u)?;
        passes.dispatch(
            KernelId::LogGuided,
            labels[pass],
            &[
                (0, ub.binding()),
                (1, mosaic.binding()),
                (2, scratch.binding()),
                (3, rgb.binding()),
            ],
            (dispatch.0, dispatch.1, 1),
        )?;
    }
    Ok(())
}

/// Records the log-guided demosaic of a mosaic that is on the device and returns the three planes as
/// a new [`GpuImage`] (device in, device out; the host wrappers [`demosaic`], [`demosaic_tiled`] and
/// [`demosaic_planned`] are a thin layer on the same passes).
///
/// `input` is the one-plane mosaic in `f32` (the planes of the plan are `f32` in every storage mode;
/// a device opened with an `f16` storage mode for the legacy kernels is refused here), `cfa` the
/// pattern at its first photosite. The output keeps the input's [`Placement`](crate::cpu::Placement).
/// Nothing is submitted.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not one plane, an empty one, or a device that is
/// not in the `f32` storage mode; the pool's errors; shader errors.
pub fn encode_demosaic(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    cfa: Cfa,
    noise: NoiseDn,
    guide_detail: i32,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if ctx.storage_mode() != StorageMode::F32 {
        return Err(GpuError::InvalidInput(format!(
            "log-guided demosaic: the device entry point needs f32 storage, this device uses {}",
            ctx.storage_mode().name()
        )));
    }
    if input.planes != 1 {
        return Err(GpuError::InvalidInput(format!(
            "log-guided demosaic: the input must be one plane, got {}",
            input.planes
        )));
    }
    let out = GpuImage::alloc(
        ctx,
        "log-guided rgb",
        input.width,
        input.height,
        3,
        input.place,
    )?;
    record(
        passes,
        &input.buf,
        &out.buf,
        (input.width, input.height),
        cfa,
        noise,
        guide_detail,
    )?;
    Ok(out)
}

/// Runs the kernel on one mosaic rectangle and reads the planes back.
fn demosaic_rect(
    ctx: &GpuContext,
    m: &Mosaic,
    noise: NoiseDn,
    guide_detail: i32,
) -> Result<RgbImage, GpuError> {
    check_mosaic(m)?;
    let mode = ctx.storage_mode();
    let (w, h) = (m.width, m.height);
    let stride = even(w);
    let plane = stride * h;
    let mut passes = PassList::new(ctx, "log-guided demosaic");
    let mosaic = passes.keep(upload_mosaic(ctx, m)?);
    let rgb = passes.keep(ctx.pool().acquire(
        "log-guided rgb",
        sample_bytes(mode, 3 * plane),
        STORAGE_OUT,
    )?);
    record(
        &mut passes,
        &mosaic,
        &rgb,
        (w, h),
        m.cfa,
        noise,
        guide_detail,
    )?;
    let read = passes.read(&rgb, 0, 3 * plane as u64 * mode.bytes_per_sample())?;
    let done = passes.submit()?.finish()?;
    let raw = image_samples(mode, done.read(read));
    let mut out = RgbImage::new(w, h);
    out.place = m.place;
    for (c, dst) in out.planes_mut().into_iter().enumerate() {
        for y in 0..h {
            let s = c * plane + y * stride;
            dst[y * w..(y + 1) * w].copy_from_slice(&raw[s..s + w]);
        }
    }
    Ok(out)
}

/// Demosaics `m` in one go: the whole buffer is one rectangle, mirrored at its edges.
///
/// `noise` sets the offset `p` (only its read-noise part is used, as on the CPU) and
/// `guide_detail` the sharpness of the guide, clamped to `-50..=100`; the product uses 0. The
/// output has the mosaic's size and [`Placement`](crate::cpu::Placement). The scratch planes need
/// 16 bytes per pixel in one storage binding: for a big frame use [`demosaic_tiled`].
///
/// This is the GPU counterpart of
/// [`demosaic_float`](crate::cpu::demosaic::log_guided::demosaic_float); see the module
/// documentation for how they differ.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty mosaic or one whose data does not match its size;
/// [`GpuError::BufferTooLarge`] and [`GpuError::OutOfBudget`] when the buffers do not fit the
/// device; shader and readback errors.
pub fn demosaic(
    ctx: &GpuContext,
    m: &Mosaic,
    noise: NoiseDn,
    guide_detail: i32,
) -> Result<RgbImage, GpuError> {
    demosaic_rect(ctx, m, noise, guide_detail)
}

/// Demosaics `m` tile by tile: cores of at most `tile_px` (rounded down to an even number), each
/// computed from a rectangle that carries [`HALO_PX`] pixels of context (clamped to the frame,
/// left and top edges even so the Bayer phase is unchanged) and pasted into the result.
///
/// Gives the same pixels as [`demosaic`] on the whole frame, up to the last bit of `f32` (the tests
/// state the bound). Tiles are processed one after the other; `cancel` is checked between them.
///
/// # Errors
///
/// [`GpuError::Cancelled`] when `cancel` fires, plus the errors of [`demosaic`].
pub fn demosaic_tiled(
    ctx: &GpuContext,
    m: &Mosaic,
    noise: NoiseDn,
    guide_detail: i32,
    tile_px: usize,
    cancel: &CancelToken,
) -> Result<RgbImage, GpuError> {
    check_mosaic(m)?;
    let plan = TilePlan::new(
        m.width,
        m.height,
        tile_px,
        &[Stage::Demosaic(crate::cpu::demosaic::Method::LogGuided)],
    );
    demosaic_planned(ctx, m, noise, guide_detail, &plan, cancel)
}

/// Like [`demosaic_tiled`] with a plan made by the caller, which must be for the size of `m` and
/// give every tile at least [`HALO_PX`] pixels of context around its core (a plan built with
/// `Stage::Demosaic(Method::LogGuided)` does).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the plan is for another size or a tile lacks context (the
/// kernel mirrors at the edge of the tile's rectangle, so missing context would show as seams, not
/// as an error); [`GpuError::Cancelled`] when `cancel` fires; plus the errors of [`demosaic`].
pub fn demosaic_planned(
    ctx: &GpuContext,
    m: &Mosaic,
    noise: NoiseDn,
    guide_detail: i32,
    plan: &TilePlan,
    cancel: &CancelToken,
) -> Result<RgbImage, GpuError> {
    check_mosaic(m)?;
    let frame = plan.frame();
    if (frame.w, frame.h) != (m.width, m.height) {
        return Err(GpuError::InvalidInput(format!(
            "tile plan is for {}x{}, the mosaic is {}x{}",
            frame.w, frame.h, m.width, m.height
        )));
    }
    for tile in plan.tiles() {
        let needed = tile.core.grow(Halo::uniform(HALO_PX), frame);
        if !tile.compute.contains(&needed) {
            return Err(GpuError::InvalidInput(format!(
                "tile {} computes {:?}, but its core {:?} needs {HALO_PX} px of context ({needed:?})",
                tile.index, tile.compute, tile.core
            )));
        }
    }
    let mut out = RgbImage::new(m.width, m.height);
    out.place = m.place;
    for &tile in plan.tiles() {
        cancel.check()?;
        let sub = m.extract(tile.compute);
        let img = demosaic_rect(ctx, &sub, noise, guide_detail)?;
        out.paste(&img, tile.core_in_compute(), tile.core.x, tile.core.y);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    /// Metal's compiler runs with fast-math, which rewrites `x * (a / b)` into `(x * a) / b`: with
    /// `x` and `a` both on the working scale of a very bright frame (A up to 2e35) the product
    /// overflows although every factor and the result are finite. Compiled with clang
    /// `-O2 -ffast-math`, the reconstruction of this shader gave infinity from a scale of about 3e21
    /// on and the same numbers as IEEE with a barrier after each quotient (lavapipe does not
    /// rewrite, so only the text of the shader can pin the barriers here).
    #[test]
    fn every_quotient_of_the_reconstruction_is_behind_a_barrier() {
        let body = include_str!("../shaders/log_guided.wgsl");
        for line in [
            "let q = lg_opaque(cc / (wh * hc + wv * vc));",
            "other = (sa * lg_opaque(cc / sc)) / s;",
            "chroma = (lg_opaque(cc / hc) * ha) / s;",
            "other = (lg_opaque(cc / vc) * va) / s;",
            "return bitcast<f32>(bitcast<u32>(x) ^ u.ctl.y);",
        ] {
            assert!(body.contains(line), "the shader lost `{line}`");
        }
    }
}

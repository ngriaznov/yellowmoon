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
//!
//! Not wired into the render path: the raw stages still run the CPU reference. The entry points here
//! upload, run and read back, for tests and for the later decision on the preview path.

use bytemuck::{Pod, Zeroable};

use super::raw::RgbPlanes;
use super::shader::KernelId;
use super::util::{
    GROUP, STORAGE_IN, STORAGE_OUT, STORAGE_SCRATCH, bind, even, groups, image_bytes,
    image_samples, sample_bytes, submit_with_readback, uniform_buffer,
};
use crate::cpu::demosaic::log_guided::{HALO_PX, derive};
use crate::cpu::noise::WORKING_SCALE;
use crate::cpu::tiling::{Halo, Stage};
use crate::cpu::{Mosaic, NoiseDn, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
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
    /// The pass.
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
    ctx.queue.write_buffer(&buf, 0, &bytes);
    Ok(buf)
}

/// Encodes the four dispatches over `m` (already uploaded as `mosaic`) and returns the encoder,
/// not yet submitted, with the planes it writes. The planes are allocated with `rgb_usage`.
fn encode(
    ctx: &GpuContext,
    m: &Mosaic,
    mosaic: PooledBuffer,
    noise: NoiseDn,
    guide_detail: i32,
    rgb_usage: wgpu::BufferUsages,
) -> Result<(wgpu::CommandEncoder, RgbPlanes), GpuError> {
    let mode = ctx.storage_mode();
    let (w, h) = (m.width, m.height);
    let stride = even(w);
    let plane = stride * h;
    // The float form: no headroom, p from the read noise at H = 1 (spec 2.2).
    let d = derive(noise, guide_detail, 0);
    let tap = |i: usize| d.kernel[i] as f32 / KERNEL_SUM;
    let base = LogGuidedU {
        dims: [w as u32, h as u32, stride as u32, plane as u32],
        cfa: std::array::from_fn(|p| m.cfa.color_at(p & 1, p >> 1).plane() as u32),
        taps: [tap(0), tap(1), tap(2), 0.0],
        misc: [d.p as f32, WORKING_SCALE as f32, 0.0, 0.0],
        ctl: [0; 4],
    };

    let scratch = ctx.pool().acquire(
        "log-guided scratch",
        SCRATCH_PLANES as u64 * plane as u64 * 4,
        STORAGE_SCRATCH,
    )?;
    let rgb = ctx
        .pool()
        .acquire("log-guided rgb", sample_bytes(mode, 3 * plane), rgb_usage)?;
    let pipeline = ctx.kernels().pipeline(ctx, KernelId::LogGuided)?;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("log-guided demosaic"),
        });
    let mut held = Vec::with_capacity(6);
    let per_pixel = (
        w.div_ceil(GROUP as usize) as u32,
        h.div_ceil(GROUP as usize) as u32,
    );
    let per_pair = groups(stride, h);
    for (pass, dispatch) in [per_pixel, per_pixel, per_pixel, per_pair]
        .into_iter()
        .enumerate()
    {
        let mut u = base;
        u.ctl[0] = pass as u32;
        let ub = uniform_buffer(ctx, "log-guided uniform", &u)?;
        let group = bind(
            ctx,
            &pipeline,
            "log-guided",
            &[
                (0, ub.binding()),
                (1, mosaic.binding()),
                (2, scratch.binding()),
                (3, rgb.binding()),
            ],
        );
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(
                    [
                        "log-guided Lh",
                        "log-guided C",
                        "log-guided SH SV",
                        "log-guided rgb",
                    ][pass],
                ),
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &group, &[]);
            cp.dispatch_workgroups(dispatch.0, dispatch.1, 1);
        }
        held.push(ub);
    }
    held.push(mosaic);
    held.push(scratch);
    Ok((
        encoder,
        RgbPlanes {
            rgb,
            stride,
            plane,
            held,
        },
    ))
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
    let mosaic = upload_mosaic(ctx, m)?;
    let (encoder, planes) = encode(ctx, m, mosaic, noise, guide_detail, STORAGE_OUT)?;
    let bytes = 3 * planes.plane as u64 * mode.bytes_per_sample();
    let readback = submit_with_readback(ctx, encoder, &planes.rgb, bytes)?;
    let raw = image_samples(mode, &readback.finish(ctx)?);
    let (w, h) = (m.width, m.height);
    let mut out = RgbImage::new(w, h);
    out.place = m.place;
    for (c, dst) in out.planes_mut().into_iter().enumerate() {
        for y in 0..h {
            let s = c * planes.plane + y * planes.stride;
            dst[y * w..(y + 1) * w].copy_from_slice(&raw[s..s + w]);
        }
    }
    drop(planes);
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

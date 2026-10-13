//! The M0-07 spike chain on the GPU: R1 normalise and R4 gains, bilinear demosaic, a fixed 3x3
//! matrix and the sRGB transfer, tile by tile with halos (`docs/15-milestones.md` M0-07).
//!
//! Three compute passes per tile, all over the tile's *compute* rectangle except the last, which
//! runs over the *core* only and writes the dense result:
//!
//! 1. `spike_normalize`: raw `u16` (two per word) to the normalised, gain-balanced mosaic in f16.
//! 2. `spike_demosaic`: bilinear demosaic to three f16 planes.
//! 3. `spike_matrix`: matrix, sRGB encode, clamp; written as f32 planes, RGBA8 or u16 planes.
//!
//! The CPU reference is the real classical chain ([`crate::cpu::chain::Chain`] with every stage but
//! normalise, gains and bilinear switched off) followed by the same matrix and
//! `ym_color::encode::srgb_encode`: [`cpu_reference`]. GPU and CPU agree within 1e-3 (the f16
//! storage of the mosaic and the demosaiced planes accounts for nearly all of it).
//!
//! Tiles are uploaded, computed and read back with up to `depth` tiles in flight; every buffer
//! comes from the pool, so [`SpikeStats`] reports the peak GPU memory of the run.

use std::collections::VecDeque;

use ym_color::consts::linalg::Mat3;
use ym_color::encode::srgb_encode;

use super::raw::{RawParams, SpikeU, check_plan, run_front};
use super::shader::KernelId;
use super::util::{
    OutKind, PixelData, Readback, STORAGE_OUT, bind, even, groups, submit_with_readback,
    uniform_buffer, unpack,
};
use crate::cpu::chain::{Chain, RawFrame};
use crate::cpu::demosaic::Method;
use crate::cpu::detail::NoiseReduction;
use crate::cpu::normalize::Levels;
use crate::cpu::tiling::Rect;
use crate::cpu::{Cfa, RgbImage};
use crate::device::{GpuContext, GpuError, StorageMode};
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::tiles::{Tile, TilePlan};

/// Parameters of the spike chain.
#[derive(Clone, Debug)]
pub struct SpikeParams {
    /// The pattern at the frame's first photosite.
    pub cfa: Cfa,
    /// R1 black and white levels per CFA position.
    pub levels: Levels,
    /// R4 gains (red, green, blue).
    pub gains: [f32; 3],
    /// The fixed matrix from balanced camera RGB to linear display RGB.
    pub matrix: Mat3,
}

impl SpikeParams {
    /// The raw front end's part of the parameters.
    pub fn raw(&self) -> RawParams {
        RawParams {
            cfa: self.cfa,
            levels: self.levels,
            gains: self.gains,
        }
    }
}

/// What the last kernel writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Three f32 planes of values in [0, 1].
    F32Planar,
    /// 8-bit RGBA, rounded (no dither), alpha 255.
    Rgba8,
    /// Three u16 planes, `round(v * 65535)`.
    U16Planar,
}

impl OutputFormat {
    fn out_kind(self) -> OutKind {
        match self {
            OutputFormat::F32Planar => OutKind::F32Planar,
            OutputFormat::Rgba8 => OutKind::Rgba8,
            OutputFormat::U16Planar => OutKind::U16Planar,
        }
    }

    /// Bytes of device output per core pixel.
    pub const fn bytes_per_pixel(self) -> u64 {
        match self {
            OutputFormat::F32Planar => 12,
            OutputFormat::Rgba8 => 4,
            OutputFormat::U16Planar => 6,
        }
    }
}

/// Pixels of a finished tile (see [`PixelData`]).
pub type TileData = PixelData;

/// A finished tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TileOutput {
    /// The tile (its `core` is where the data belongs in the frame).
    pub tile: Tile,
    /// Core width in pixels.
    pub width: usize,
    /// Core height in pixels.
    pub height: usize,
    /// The pixels.
    pub data: TileData,
}

/// What a run measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpikeStats {
    /// Tiles processed.
    pub tiles: usize,
    /// Peak bytes of all live GPU buffers during the run (see [`crate::pool`]).
    pub peak_allocated_bytes: u64,
    /// Peak bytes held by running tiles.
    pub peak_in_use_bytes: u64,
    /// Buffers created from the device (the rest were reused).
    pub buffers_created: u64,
    /// Buffers reused from the pool.
    pub reuses: u64,
    /// The driver-side allocator total at the end of the run, where wgpu can report it.
    pub allocator_total_bytes: Option<u64>,
}

/// Device bytes per compute-rectangle pixel across the buffers of one tile in flight: raw `u16`,
/// the mosaic, three planes, plus the output and its staging copy (counted at core size, which is
/// a bit smaller than the compute rectangle, so this is an upper bound).
pub fn bytes_per_pixel(mode: StorageMode, format: OutputFormat) -> u64 {
    2 + mode.bytes_per_sample() * 4 + 2 * format.bytes_per_pixel()
}

fn row4(r: &[f64; 3]) -> [f32; 4] {
    [r[0] as f32, r[1] as f32, r[2] as f32, 0.0]
}

/// Everything held while a tile is on the GPU.
struct InFlight {
    tile: Tile,
    core_w: usize,
    core_h: usize,
    out_stride: usize,
    readback: Readback,
    _held: Vec<PooledBuffer>,
}

fn submit_tile(
    ctx: &GpuContext,
    raw: &RawFrame<'_>,
    params: &SpikeParams,
    format: OutputFormat,
    tile: Tile,
    base: &SpikeU,
) -> Result<InFlight, GpuError> {
    let front = run_front(ctx, raw, &params.raw(), base, &tile)?;
    let core_w = tile.core.w;
    let core_h = tile.core.h;
    let out_stride = even(core_w);
    let out_plane = out_stride * core_h;
    let (dx, dy) = tile.core_offset();
    let out_bytes = format.out_kind().device_bytes(out_stride, core_h);
    let out = ctx.pool().acquire("tile out", out_bytes, STORAGE_OUT)?;

    let mut u3 = *base;
    u3.dims = [
        core_w as u32,
        core_h as u32,
        out_stride as u32,
        out_plane as u32,
    ];
    u3.src = [
        front.stride as u32,
        front.plane as u32,
        dx as u32,
        dy as u32,
    ];
    u3.parity = [0, 0, format.out_kind().code(), 0];
    u3.m0 = row4(&params.matrix[0]);
    u3.m1 = row4(&params.matrix[1]);
    u3.m2 = row4(&params.matrix[2]);
    let ub3 = uniform_buffer(ctx, "spike u3", &u3)?;
    let p3 = ctx.kernels().pipeline(ctx, KernelId::SpikeMatrix)?;
    let g3 = bind(
        ctx,
        &p3,
        "spike matrix",
        &[
            (0, ub3.binding()),
            (1, front.rgb.binding()),
            (2, out.binding()),
        ],
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("spike matrix"),
        });
    {
        let (gx, gy) = groups(out_stride, core_h);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spike matrix"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&p3);
        pass.set_bind_group(0, &g3, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    let readback = submit_with_readback(ctx, encoder, &out, out_bytes)?;
    let mut held = front.held;
    held.extend([front.rgb, out, ub3]);
    Ok(InFlight {
        tile,
        core_w,
        core_h,
        out_stride,
        readback,
        _held: held,
    })
}

fn finish_tile(
    ctx: &GpuContext,
    format: OutputFormat,
    job: InFlight,
) -> Result<TileOutput, GpuError> {
    let InFlight {
        tile,
        core_w,
        core_h,
        out_stride,
        readback,
        _held,
    } = job;
    let bytes = readback.finish(ctx)?;
    let data = unpack(format.out_kind(), &bytes, core_w, core_h, out_stride);
    Ok(TileOutput {
        tile,
        width: core_w,
        height: core_h,
        data,
    })
}

/// Runs the spike chain over every tile of `plan` and hands each finished tile to `sink` in raster
/// order, so a streaming encoder can take rows as soon as a tile row is complete.
///
/// `depth` tiles are kept in flight (upload and compute of the next tile overlap the readback of
/// the previous one; 1 is fully sequential). `cancel` is polled between tiles.
///
/// # Errors
///
/// [`GpuError::Cancelled`] when `cancel` fires, [`GpuError::InvalidInput`] when the plan does not
/// match the frame, and any allocation, shader or readback error.
#[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
pub fn run_spike(
    ctx: &GpuContext,
    raw: &RawFrame<'_>,
    params: &SpikeParams,
    plan: &TilePlan,
    format: OutputFormat,
    depth: usize,
    cancel: &CancelToken,
    sink: &mut dyn FnMut(TileOutput) -> Result<(), GpuError>,
) -> Result<SpikeStats, GpuError> {
    check_plan(plan, raw, params.cfa)?;
    params.levels.validate()?;
    let uniform = SpikeU::from_raw(&params.raw());
    ctx.pool().reset_peaks();
    let before = ctx.pool().stats();
    let mut queue: VecDeque<InFlight> = VecDeque::new();
    let mut tiles = 0;
    let depth = depth.max(1);
    for &tile in plan.tiles() {
        cancel.check()?;
        queue.push_back(submit_tile(ctx, raw, params, format, tile, &uniform)?);
        if queue.len() >= depth
            && let Some(job) = queue.pop_front()
        {
            sink(finish_tile(ctx, format, job)?)?;
            tiles += 1;
        }
    }
    while let Some(job) = queue.pop_front() {
        sink(finish_tile(ctx, format, job)?)?;
        tiles += 1;
    }
    let after = ctx.pool().stats();
    Ok(SpikeStats {
        tiles,
        peak_allocated_bytes: after.peak_allocated_bytes,
        peak_in_use_bytes: after.peak_in_use_bytes,
        buffers_created: after.buffers_created - before.buffers_created,
        reuses: after.reuses - before.reuses,
        allocator_total_bytes: ctx
            .device
            .generate_allocator_report()
            .map(|r| r.total_allocated_bytes),
    })
}

/// The CPU reference of the spike chain for `rect` of the frame (frame coordinates): the classical
/// chain with bilinear demosaic, then the matrix and the sRGB transfer, as planar f32 in [0, 1].
///
/// # Errors
///
/// Whatever the CPU stages report for inconsistent inputs.
pub fn cpu_reference(
    raw: &RawFrame<'_>,
    params: &SpikeParams,
    rect: Rect,
) -> Result<RgbImage, GpuError> {
    let chain = Chain {
        levels: params.levels,
        vignette: None,
        gains: params.gains,
        highlight_recovery: false,
        method: Method::Bilinear,
        denoise: None,
        noise_reduction: NoiseReduction {
            luminance: 0.0,
            color: 0.0,
            detail: 0.0,
        },
        iso: 100.0,
        sigma: 0.0,
        sigma_map: None,
        noise: None,
        sharpen: None,
        lens: None,
    };
    let prepared = chain.prepare(raw)?;
    let linear = chain.run_tile(raw, &prepared, rect)?;
    let mut out = RgbImage::new(linear.width, linear.height);
    let n = linear.plane_len();
    for i in 0..n {
        let rgb = [
            f64::from(linear.data[i]),
            f64::from(linear.data[n + i]),
            f64::from(linear.data[2 * n + i]),
        ];
        let lin = ym_color::consts::linalg::mul_vec(&params.matrix, rgb);
        for (c, v) in lin.iter().enumerate() {
            out.data[c * n + i] = srgb_encode(*v).clamp(0.0, 1.0) as f32;
        }
    }
    Ok(out)
}

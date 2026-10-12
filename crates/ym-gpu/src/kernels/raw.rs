//! The raw front end on the GPU: R1 normalise, R4 gains and the bilinear demosaic, from the raw
//! `u16` tile to three f16 planes that stay on the device, plus the raw-to-display driver that
//! feeds those planes straight into the colour kernel ([`run_raw_to_display`]) and the row
//! streaming helpers shared by the drivers.

use bytemuck::{Pod, Zeroable};

use super::color::{ColorImage, ColorOutput, OutWindow, PreparedChain, SrcWindow};
use super::shader::KernelId;
use super::util::{
    PixelData, STORAGE_IN, STORAGE_SCRATCH, bind, even, groups, sample_bytes, uniform_buffer,
};
use crate::cpu::Cfa;
use crate::cpu::chain::RawFrame;
use crate::cpu::demosaic::Method;
use crate::cpu::normalize::Levels;
use crate::cpu::tiling::Halo;
use crate::device::{GpuContext, GpuError, StorageMode};
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::tiles::{RowCollector, Tile, TilePlan};

/// What the raw front end needs: the pattern, R1's levels and R4's gains.
#[derive(Clone, Debug)]
pub struct RawParams {
    /// The pattern at the frame's first photosite.
    pub cfa: Cfa,
    /// R1 black and white levels per CFA position.
    pub levels: Levels,
    /// R4 gains (red, green, blue).
    pub gains: [f32; 3],
}

/// The uniform block shared by the three spike shaders (and so by the front end).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SpikeU {
    pub dims: [u32; 4],
    pub src: [u32; 4],
    pub parity: [u32; 4],
    pub black: [f32; 4],
    pub inv_range: [f32; 4],
    pub gain: [f32; 4],
    pub cfa: [u32; 4],
    pub m0: [f32; 4],
    pub m1: [f32; 4],
    pub m2: [f32; 4],
}

impl SpikeU {
    /// The per-frame part: levels and gains by CFA position.
    pub(crate) fn from_raw(params: &RawParams) -> SpikeU {
        let inv: [f32; 4] =
            std::array::from_fn(|i| 1.0 / (params.levels.white[i] - params.levels.black[i]));
        let gain: [f32; 4] =
            std::array::from_fn(|p| params.gains[params.cfa.color_at(p & 1, p >> 1).plane()]);
        SpikeU {
            dims: [0; 4],
            src: [0; 4],
            parity: [0; 4],
            black: params.levels.black,
            inv_range: inv,
            gain,
            cfa: [0; 4],
            m0: [0.0; 4],
            m1: [0.0; 4],
            m2: [0.0; 4],
        }
    }
}

/// Demosaiced planes of one tile, on the device, in the storage mode, with an even row stride.
pub(crate) struct RgbPlanes {
    pub rgb: PooledBuffer,
    pub stride: usize,
    pub plane: usize,
    /// Buffers the submitted work reads; keep them alive until the tile's readback finished.
    pub held: Vec<PooledBuffer>,
}

/// Uploads the tile's raw rectangle and runs normalise + gains and the bilinear demosaic over its
/// compute rectangle. The work is submitted; nothing is read back.
pub(crate) fn run_front(
    ctx: &GpuContext,
    raw: &RawFrame<'_>,
    params: &RawParams,
    base: &SpikeU,
    tile: &Tile,
) -> Result<RgbPlanes, GpuError> {
    let mode: StorageMode = ctx.storage_mode();
    let c = tile.compute;
    let stride = even(c.w);
    let plane = stride * c.h;

    // Raw tile: u16 rows padded to the even stride.
    let mut words: Vec<u16> = vec![0; plane];
    for y in 0..c.h {
        let src = (c.y + y) * raw.width + c.x;
        words[y * stride..y * stride + c.w].copy_from_slice(&raw.data[src..src + c.w]);
    }
    let raw_buf = ctx
        .pool()
        .acquire("raw tile", plane as u64 * 2, STORAGE_IN)?;
    ctx.queue
        .write_buffer(&raw_buf, 0, bytemuck::cast_slice(&words));
    let mosaic = ctx
        .pool()
        .acquire("mosaic", sample_bytes(mode, plane), STORAGE_SCRATCH)?;
    let rgb = ctx
        .pool()
        .acquire("rgb", sample_bytes(mode, 3 * plane), STORAGE_SCRATCH)?;

    let tile_cfa = params.cfa.shifted(c.x, c.y);
    let mut u1 = *base;
    u1.dims = [c.w as u32, c.h as u32, stride as u32, plane as u32];
    u1.parity = [(c.x & 1) as u32, (c.y & 1) as u32, 0, 0];
    let mut u2 = u1;
    u2.cfa = std::array::from_fn(|p| tile_cfa.color_at(p & 1, p >> 1).plane() as u32);
    let ub1 = uniform_buffer(ctx, "normalize uniform", &u1)?;
    let ub2 = uniform_buffer(ctx, "demosaic uniform", &u2)?;

    let k = ctx.kernels();
    let p1 = k.pipeline(ctx, KernelId::SpikeNormalize)?;
    let p2 = k.pipeline(ctx, KernelId::SpikeDemosaic)?;
    let g1 = bind(
        ctx,
        &p1,
        "normalize",
        &[
            (0, ub1.binding()),
            (1, raw_buf.binding()),
            (2, mosaic.binding()),
        ],
    );
    let g2 = bind(
        ctx,
        &p2,
        "demosaic",
        &[
            (0, ub2.binding()),
            (1, mosaic.binding()),
            (2, rgb.binding()),
        ],
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("raw front end"),
        });
    let (gx, gy) = groups(stride, c.h);
    for (label, pipeline, group) in [("normalize", &p1, &g1), ("demosaic", &p2, &g2)] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit([encoder.finish()]);
    Ok(RgbPlanes {
        rgb,
        stride,
        plane,
        held: vec![raw_buf, mosaic, ub1, ub2],
    })
}

/// What a raw-to-display run measured (the pool's peak counters, see [`crate::pool`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    /// Tiles processed.
    pub tiles: usize,
    /// Peak bytes of all live GPU buffers during the run.
    pub peak_allocated_bytes: u64,
    /// Peak bytes held by running tiles.
    pub peak_in_use_bytes: u64,
}

/// Runs the raw front end and the colour chain over every tile of `plan`: raw `u16` in, display
/// pixels out, the demosaiced planes never leaving the device. `sink` receives the tiles in raster
/// order (use [`RowStreamer`] to turn them into full-width rows). `depth` tiles are in flight.
///
/// # Errors
///
/// [`GpuError::Cancelled`] when `cancel` fires, [`GpuError::InvalidInput`] when the plan does not
/// match the frame, and allocation, shader or readback errors.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_to_display(
    ctx: &GpuContext,
    raw: &RawFrame<'_>,
    params: &RawParams,
    chain: &PreparedChain,
    plan: &TilePlan,
    output: ColorOutput,
    depth: usize,
    cancel: &CancelToken,
    sink: &mut dyn FnMut(Tile, ColorImage) -> Result<(), GpuError>,
) -> Result<PipelineStats, GpuError> {
    check_plan(plan, raw, params.cfa)?;
    params.levels.validate()?;
    let base = SpikeU::from_raw(params);
    ctx.pool().reset_peaks();
    let mut queue = std::collections::VecDeque::new();
    let depth = depth.max(1);
    let mut tiles = 0;
    for &tile in plan.tiles() {
        cancel.check()?;
        let front = run_front(ctx, raw, params, &base, &tile)?;
        let (dx, dy) = tile.core_offset();
        let pending = chain.dispatch(
            ctx,
            &front.rgb,
            &SrcWindow {
                stride: front.stride,
                plane: front.plane,
                offset: (dx, dy),
            },
            &OutWindow {
                width: tile.core.w,
                height: tile.core.h,
                origin: (tile.core.x, tile.core.y),
            },
            output,
        )?;
        queue.push_back((tile, pending, front));
        if queue.len() >= depth
            && let Some((t, p, _front)) = queue.pop_front()
        {
            sink(t, p.finish(ctx)?)?;
            tiles += 1;
        }
    }
    while let Some((t, p, _front)) = queue.pop_front() {
        sink(t, p.finish(ctx)?)?;
        tiles += 1;
    }
    let stats = ctx.pool().stats();
    Ok(PipelineStats {
        tiles,
        peak_allocated_bytes: stats.peak_allocated_bytes,
        peak_in_use_bytes: stats.peak_in_use_bytes,
    })
}

/// Context the bilinear demosaic needs around a tile's core, in pixels (the CPU reference's own
/// measured receptive field).
const FRONT_HALO_PX: usize = Method::Bilinear.halo();

/// Refuses a run that would silently produce wrong pixels: a plan for another frame size, a raw
/// frame whose pattern differs from the one the kernels are told (the GPU path reads
/// `RawParams::cfa`, the CPU reference `RawFrame::cfa`), and tiles whose compute rectangle lacks the
/// halo of the demosaic (the kernels mirror at the edge of the rectangle, so a missing halo shows
/// as seams at the tile borders, not as an error).
pub(crate) fn check_plan(plan: &TilePlan, raw: &RawFrame<'_>, cfa: Cfa) -> Result<(), GpuError> {
    if (plan.frame().w, plan.frame().h) != (raw.width, raw.height)
        || raw.data.len() != raw.width * raw.height
    {
        return Err(GpuError::InvalidInput(format!(
            "tile plan is for {}x{}, the raw frame is {}x{} with {} samples",
            plan.frame().w,
            plan.frame().h,
            raw.width,
            raw.height,
            raw.data.len()
        )));
    }
    if raw.cfa != cfa {
        return Err(GpuError::InvalidInput(format!(
            "the raw frame has the pattern {} but the parameters say {}",
            raw.cfa.name(),
            cfa.name()
        )));
    }
    let frame = plan.frame();
    for tile in plan.tiles() {
        let needed = tile.core.grow(Halo::uniform(FRONT_HALO_PX), frame);
        if !tile.compute.contains(&needed) {
            return Err(GpuError::InvalidInput(format!(
                "tile {} computes {:?}, but its core {:?} needs {FRONT_HALO_PX} px of halo ({needed:?}); \
                 build the plan with the stages of the chain (`TilePlan::new`)",
                tile.index, tile.compute, tile.core
            )));
        }
    }
    Ok(())
}

/// A full-width band of the output: `height` rows from row `y0`, dense, in the layout of its
/// [`PixelData`] kind (planar f32 or u16 planes of `width * height`, or interleaved RGBA8).
#[derive(Clone, Debug, PartialEq)]
pub struct RowBand {
    /// First row of the band in the frame.
    pub y0: usize,
    /// Frame width.
    pub width: usize,
    /// Rows in the band.
    pub height: usize,
    /// The pixels.
    pub data: PixelData,
}

/// Collects finished tiles and hands the encoder one tile row at a time, as soon as the last tile
/// of the row is in (`docs/01`: "export runs tile by tile and writes rows to the encoder as soon as
/// a tile row is complete").
pub struct RowStreamer<'a> {
    frame_w: usize,
    collector: RowCollector<PixelData>,
    sink: &'a mut dyn FnMut(RowBand) -> Result<(), GpuError>,
}

impl<'a> RowStreamer<'a> {
    /// A streamer for `plan` that calls `sink` once per completed tile row.
    pub fn new(plan: &TilePlan, sink: &'a mut dyn FnMut(RowBand) -> Result<(), GpuError>) -> Self {
        Self {
            frame_w: plan.frame().w,
            collector: RowCollector::new(plan),
            sink,
        }
    }

    /// Adds a finished tile (its core, dense). Calls the sink when this completes a tile row.
    ///
    /// # Errors
    ///
    /// Whatever the sink returns.
    pub fn push(&mut self, tile: Tile, data: PixelData) -> Result<(), GpuError> {
        if let Some(row) = self.collector.push(tile, data) {
            (self.sink)(compose_row(self.frame_w, &row))?;
        }
        Ok(())
    }

    /// Tile rows not yet released.
    pub fn rows_pending(&self) -> usize {
        self.collector.rows_pending()
    }
}

/// Assembles the tiles of one row (left to right) into a full-width band.
fn compose_row(frame_w: usize, row: &[(Tile, PixelData)]) -> RowBand {
    let first = row[0].0.core;
    let (y0, h) = (first.y, first.h);
    let kind = &row[0].1;
    let data = match kind {
        PixelData::Rgba8(_) => {
            let mut out = vec![0u8; frame_w * h * 4];
            for (tile, d) in row {
                let PixelData::Rgba8(src) = d else { continue };
                let c = tile.core;
                for y in 0..c.h {
                    let dst = (y * frame_w + c.x) * 4;
                    out[dst..dst + c.w * 4].copy_from_slice(&src[y * c.w * 4..][..c.w * 4]);
                }
            }
            PixelData::Rgba8(out)
        }
        PixelData::F32Planar(_) => {
            let mut out = vec![0f32; 3 * frame_w * h];
            for (tile, d) in row {
                let PixelData::F32Planar(src) = d else {
                    continue;
                };
                paste_planar(&mut out, src, frame_w, h, tile);
            }
            PixelData::F32Planar(out)
        }
        PixelData::U16Planar(_) => {
            let mut out = vec![0u16; 3 * frame_w * h];
            for (tile, d) in row {
                let PixelData::U16Planar(src) = d else {
                    continue;
                };
                paste_planar(&mut out, src, frame_w, h, tile);
            }
            PixelData::U16Planar(out)
        }
    };
    RowBand {
        y0,
        width: frame_w,
        height: h,
        data,
    }
}

fn paste_planar<T: Copy>(out: &mut [T], src: &[T], frame_w: usize, band_h: usize, tile: &Tile) {
    let c = tile.core;
    for ch in 0..3 {
        for y in 0..c.h {
            let dst = ch * frame_w * band_h + y * frame_w + c.x;
            let s = ch * c.w * c.h + y * c.w;
            out[dst..dst + c.w].copy_from_slice(&src[s..s + c.w]);
        }
    }
}

/// The tile size to use for this device, frame chain and output: the largest of
/// [`crate::tiles::TILE_SIZES`] whose working set (two tiles in flight) fits in half the memory
/// budget and whose three image planes fit one storage-buffer binding.
pub fn recommended_tile_px(ctx: &GpuContext, bytes_per_pixel: u64, halo_px: usize) -> usize {
    let info = ctx.info();
    let by_budget = crate::tiles::tile_size_for_budget(
        info.memory_budget_bytes,
        bytes_per_pixel,
        halo_px,
        2,
        &crate::tiles::TILE_SIZES,
    );
    // Three planes of the compute rectangle must fit one binding.
    let per_px = 3 * ctx.storage_mode().bytes_per_sample().max(2);
    let mut tile = by_budget;
    while tile > 256 {
        let side = (tile + 2 * halo_px) as u64;
        if side * side * per_px <= info.limits.max_storage_buffer_binding_size {
            break;
        }
        tile /= 2;
    }
    tile
}

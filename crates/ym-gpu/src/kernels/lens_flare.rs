//! L3, the lens's highlight optics, on the device (`docs/21` section 4, L3; `docs/17-gpu.md` section 11).
//!
//! The CPU reference is `ym_lens::l3` (reached through [`ym_lens::apply_flares`]): for each output pixel
//! `out = (1 - a) Z + F (core + tail + items)`, where `core` is the glow core gathered from the restored
//! light within 8 px, `tail` the tail plane (glow tail and veil, made once per photo) read with a
//! Catmull-Rom curve, and `items` the ghosts, starbursts and streaks drawn analytically from the source
//! list. This module is the same computation in `f32`.
//!
//! # Passes
//!
//! `lens_flare_restore` makes the restored light (the clip test's boost) of the rectangle the core reads;
//! `lens_flare` sums the terms of every pixel and mixes them with `Z`.
//!
//! # Data
//!
//! Three kinds, by how long they live ([`ym_lens::device`]): the photo's [`FlarePlaneBuffer`] (the tail
//! plane and the spike tables, uploaded once), the lens's [`FlareLensBuffer`] (the rim, the aperture, the
//! normal tail's table) and the plan shape's [`FlareScaleBuffers`] (the core's taps, the items and their
//! bins at one scale). None of them depends on Flares: the amounts of the sum are the uniform's
//! (`ym_lens::device::FlareMix`), so a Flares move uploads nothing.
//!
//! # Positions
//!
//! D4 and D5 of `docs/17-gpu.md`: a pixel is its absolute column and row, its field coordinates
//! `(i + 0.5 - w / 2) / H` in `f32`; the items a pixel receives are those of its bin of the absolute pixel
//! grid, in list order, so a tile and the whole frame sum the same terms in the same order.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ym_lens::consts::{RIM_SAMPLES, SPIKE_SIZE};
use ym_lens::device::{
    FLARE_HEADER, FLARE_TAB_HEADER, FlareLensData, FlareMix, FlarePlaneData, FlareScaleData,
    NORMAL_TAIL_NODES,
};

use super::shader::KernelId;
use super::util::{GROUP, storage_buffer};
use crate::cpu::Placement;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// The uniform block of `lens_flare.wgsl` (`FlareU`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FlareU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    /// Frame width, height, half-diagonal in pixels, 0.
    frame: [f32; 4],
    /// `F`, `1 - a`, the spike table's radius, the tail plane's half-diagonal.
    mix: [f32; 4],
    /// Frame width, height, 1 when the axis is the buffer's y, core radius (-1: no core).
    ints: [i32; 4],
    /// The rectangle written: frame column and row, width, height.
    outr: [i32; 4],
    /// The restored light: frame column and row of its first pixel, width, height.
    resr: [i32; 4],
    /// `Z`: frame column and row of its first pixel, width, height.
    zr: [i32; 4],
    /// `Z` row stride, plane length, output row stride, plane length.
    zs: [u32; 4],
    /// Core taps offset, items offset, item count, 1 with a streak profile.
    offs: [u32; 4],
    /// Bins across, bins down, bin side in pixels, offset of the bin item lists.
    bins: [u32; 4],
    /// Tail plane width, height, offset of the spike tables in the planes buffer, 0.
    tail: [i32; 4],
    /// `g0` of each channel, `2^h - 1`.
    clip: [f32; 4],
    /// The clip test's ramp: from, to.
    ramp: [f32; 4],
    /// The grids of nodes of the ghosts of spacing 2, 4, 8, 16 and 32 px: the first node's column and row, the
    /// nodes across and down (zeros for a grid the tile does not use).
    lg: [[i32; 4]; 5],
    /// Where each grid starts in the node buffer (in vectors; the grids 0 to 3, then the grid 4, the bit set
    /// of the grids in use, and two zeros).
    lgo: [[u32; 4]; 2],
    /// The grid a node dispatch fills: its index (0 to 2), then zeros.
    lvl: [u32; 4],
}

/// The tail plane and spike tables of one photo on the device.
#[derive(Clone)]
pub struct FlarePlaneBuffer {
    buf: Arc<PooledBuffer>,
    tail_size: (usize, usize),
}

impl std::fmt::Debug for FlarePlaneBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlarePlaneBuffer")
            .field("tail_size", &self.tail_size)
            .field("bytes", &self.buf.size())
            .finish()
    }
}

impl FlarePlaneBuffer {
    /// Uploads a photo's planes.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for tables of the wrong length; the pool's errors.
    pub fn upload(ctx: &GpuContext, data: &FlarePlaneData) -> Result<Self, GpuError> {
        let (tw, th) = data.tail_size;
        if data.tail.len() != 3 * tw * th || data.spikes.len() != 3 * SPIKE_SIZE * SPIKE_SIZE {
            return Err(GpuError::InvalidInput(format!(
                "lens flare planes: {} tail and {} spike samples for a {tw} x {th} tail plane",
                data.tail.len(),
                data.spikes.len()
            )));
        }
        let mut all = Vec::with_capacity(data.tail.len() + data.spikes.len());
        all.extend_from_slice(&data.tail);
        all.extend_from_slice(&data.spikes);
        let buf = storage_buffer(ctx, "lens flare planes", bytemuck::cast_slice(&all))?;
        Ok(FlarePlaneBuffer {
            buf: Arc::new(buf),
            tail_size: data.tail_size,
        })
    }

    /// Bytes on the device.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.buf.size()
    }
}

/// The lens's rim, aperture and normal-tail tables on the device.
#[derive(Clone)]
pub struct FlareLensBuffer {
    buf: Arc<PooledBuffer>,
    key: String,
}

impl std::fmt::Debug for FlareLensBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlareLensBuffer")
            .field("key", &self.key)
            .finish()
    }
}

impl FlareLensBuffer {
    /// Uploads a lens's tables.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for tables of the wrong length; the pool's errors.
    pub fn upload(ctx: &GpuContext, data: &FlareLensData) -> Result<Self, GpuError> {
        if data.tab.len() != FLARE_TAB_HEADER + NORMAL_TAIL_NODES {
            return Err(GpuError::InvalidInput(format!(
                "lens flare tables: {} vectors, expected {}",
                data.tab.len(),
                FLARE_TAB_HEADER + NORMAL_TAIL_NODES
            )));
        }
        let buf = storage_buffer(ctx, "lens flare tables", bytemuck::cast_slice(&data.tab))?;
        Ok(FlareLensBuffer {
            buf: Arc::new(buf),
            key: data.key.clone(),
        })
    }

    /// Names the lens, its data version and the camera the tables were made for.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// One plan shape's data at one scale on the device: the core's taps, the items and their bins.
#[derive(Clone)]
pub struct FlareScaleBuffers {
    vectors: Arc<PooledBuffer>,
    bins: Arc<PooledBuffer>,
    meta: FlareScaleData,
}

impl std::fmt::Debug for FlareScaleBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlareScaleBuffers")
            .field("frame", &(self.meta.width, self.meta.height))
            .field("items", &self.meta.records)
            .field("core_radius", &self.meta.core_radius)
            .finish()
    }
}

impl FlareScaleBuffers {
    /// Uploads a scale's data (`LensPlan::flare_scale_data`).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a header that is not [`FLARE_HEADER`] vectors long; the pool's
    /// errors.
    pub fn upload(ctx: &GpuContext, data: &FlareScaleData) -> Result<Self, GpuError> {
        if data.vectors.len() < FLARE_HEADER || data.core_offset != FLARE_HEADER {
            return Err(GpuError::InvalidInput(
                "lens flare scale data: a bad header".into(),
            ));
        }
        let vectors = storage_buffer(ctx, "lens flare scale", bytemuck::cast_slice(&data.vectors))?;
        let bins = storage_buffer(ctx, "lens flare bins", bytemuck::cast_slice(&data.bins))?;
        Ok(FlareScaleBuffers {
            vectors: Arc::new(vectors),
            bins: Arc::new(bins),
            meta: data.clone(),
        })
    }

    /// Pixels of context an output rectangle needs around it: the glow core's reach.
    #[must_use]
    pub fn halo(&self) -> usize {
        self.meta.halo()
    }

    /// Bytes on the device.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.vectors.size() + self.bins.size()
    }
}

/// The first and last node (inclusive) of the grid of spacing `l` px that the pixels `[p0, p1)` read: the pixel
/// `p` lies between the nodes `floor((2 p + 1 - l) / 2 l)` and the next (`ym_lens` `node_range`).
fn node_range(p0: usize, p1: usize, l: i64) -> (i64, i64) {
    let first = (2 * p0 as i64 + 1 - l).div_euclid(2 * l);
    let last = (2 * (p1 as i64 - 1) + 1 - l).div_euclid(2 * l) + 1;
    (first, last)
}

/// The largest storage binding (bytes) the pass needs for an output of `rect` pixels with a core of
/// `halo` pixels: the restored light or an image.
#[must_use]
pub fn max_binding_bytes(rect: Rect, halo: usize) -> u64 {
    let grown = (rect.w + 2 * halo) as u64 * (rect.h + 2 * halo) as u64 * 16;
    grown
        .max(GpuImage::buffer_bytes(rect.w, rect.h, 3))
        .max(GpuImage::buffer_bytes(
            rect.w + 2 * halo,
            rect.h + 2 * halo,
            3,
        ))
}

/// Records L3 over `out` (frame coordinates at the scale of `scale`): device in, device out.
///
/// * `input` is the picture before L1 (the core reads it), three planes whose first pixel is the frame
///   pixel `input_at`; it must cover `out` grown by [`FlareScaleBuffers::halo`], clamped to the frame.
/// * `z` is the output of L2 over `out` (or any image that covers it: the first pixel of `z` is the
///   frame pixel `z_at`); with the flares off the caller does not come here.
///
/// The result is `out.w x out.h`, placed at `(out.x, out.y)` in the frame.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input or `Z` that is not three planes or does not cover what it must,
/// or an empty output or one outside the frame; the pool's errors; shader errors.
#[allow(clippy::too_many_arguments)]
pub fn encode_flare(
    passes: &mut PassList<'_>,
    planes: &FlarePlaneBuffer,
    lens: &FlareLensBuffer,
    scale: &FlareScaleBuffers,
    mix: FlareMix,
    input: &GpuImage,
    input_at: (usize, usize),
    z: &GpuImage,
    z_at: (usize, usize),
    out: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    let meta = &scale.meta;
    let (fw, fh) = (meta.width, meta.height);
    for (name, image) in [("input", input), ("Z", z)] {
        if image.planes != 3 {
            return Err(GpuError::InvalidInput(format!(
                "lens flare: {name} has {} planes, not 3",
                image.planes
            )));
        }
    }
    if out.is_empty() || out.x1() > fw || out.y1() > fh {
        return Err(GpuError::InvalidInput(format!(
            "lens flare: the output {out:?} is empty or outside the {fw} x {fh} frame"
        )));
    }
    let covers = |image: &GpuImage, at: (usize, usize), need: Rect| {
        need.x >= at.0
            && need.y >= at.1
            && need.x1() <= at.0 + image.width
            && need.y1() <= at.1 + image.height
    };
    if !covers(z, z_at, out) {
        return Err(GpuError::InvalidInput(format!(
            "lens flare: Z at {z_at:?} ({} x {}) does not cover the output {out:?}",
            z.width, z.height
        )));
    }
    let halo = meta.halo();
    let region = {
        let x = out.x.saturating_sub(halo);
        let y = out.y.saturating_sub(halo);
        let x1 = (out.x1() + halo).min(fw);
        let y1 = (out.y1() + halo).min(fh);
        Rect::new(x, y, x1 - x, y1 - y)
    };
    let has_core = !meta.vectors.is_empty() && meta.record_offset > meta.core_offset;
    if has_core && !covers(input, input_at, region) {
        return Err(GpuError::InvalidInput(format!(
            "lens flare: the input at {input_at:?} ({} x {}) does not cover the output {out:?} grown by {halo} px",
            input.width, input.height
        )));
    }
    let restored = if has_core {
        Some(passes.scratch(
            "lens flare restored light",
            region.w as u64 * region.h as u64 * 16,
        )?)
    } else {
        None
    };
    let dst = GpuImage::alloc(
        ctx,
        "lens flare",
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
    // The grids of nodes the slowly varying ghosts are drawn on, over the nodes this tile's pixels read.
    let mut grids = [[0i32; 4]; 5];
    let mut grid_offsets = [[0u32; 4]; 2];
    let mut grid_set = 0u32;
    let mut node_vectors = 0u64;
    for (li, on) in meta.coarse.iter().enumerate() {
        if !*on {
            continue;
        }
        let l = 2i64 << li;
        let (j0, j1) = node_range(out.x, out.x1(), l);
        let (k0, k1) = node_range(out.y, out.y1(), l);
        let (nx, ny) = (j1 - j0 + 1, k1 - k0 + 1);
        grids[li] = [j0 as i32, k0 as i32, nx as i32, ny as i32];
        grid_offsets[li / 4][li % 4] = node_vectors as u32;
        grid_set |= 1 << li;
        node_vectors += (nx * ny) as u64;
    }
    // Always a buffer to bind: the main pass declares the slot.
    let nodes = passes.scratch("lens flare nodes", node_vectors.max(1) * 16)?;
    let (tw, th) = planes.tail_size;
    let nodes_after = (3 * tw * th) as i32;
    let mut u = FlareU::zeroed();
    u.frame = [fw as f32, fh as f32, meta.half_diag as f32, 0.0];
    u.mix = [
        mix.f as f32,
        mix.keep as f32,
        meta.spike_radius as f32,
        (0.5 * ((tw * tw + th * th) as f64).sqrt()) as f32,
    ];
    u.ints = [
        fw as i32,
        fh as i32,
        i32::from(meta.axis_y),
        if has_core {
            meta.core_radius as i32
        } else {
            -1
        },
    ];
    u.outr = [out.x as i32, out.y as i32, out.w as i32, out.h as i32];
    u.resr = [
        region.x as i32,
        region.y as i32,
        region.w as i32,
        region.h as i32,
    ];
    u.zr = [
        z_at.0 as i32,
        z_at.1 as i32,
        z.width as i32,
        z.height as i32,
    ];
    u.zs = [
        z.stride as u32,
        z.plane_len() as u32,
        dst.stride as u32,
        dst.plane_len() as u32,
    ];
    u.offs = [
        meta.core_offset as u32,
        meta.record_offset as u32,
        meta.records as u32,
        u32::from(meta.has_streak),
    ];
    let nb = (meta.bins_x * meta.bins_y) as u32;
    u.bins = [
        meta.bins_x as u32,
        meta.bins_y as u32,
        meta.bin_px as u32,
        nb + 1,
    ];
    u.tail = [tw as i32, th as i32, nodes_after, 0];
    u.clip = [
        meta.restore.clip[0],
        meta.restore.clip[1],
        meta.restore.clip[2],
        meta.restore.boost,
    ];
    u.ramp = [meta.restore.ramp.0, meta.restore.ramp.1, 0.0, 0.0];
    u.lg = grids;
    grid_offsets[1][1] = grid_set;
    u.lgo = grid_offsets;

    if let Some(res) = &restored {
        // The restored light reads the input (binding 2 of the restore pass is the input's planes).
        let mut ur = u;
        ur.zr = [
            input_at.0 as i32,
            input_at.1 as i32,
            input.width as i32,
            input.height as i32,
        ];
        ur.zs[0] = input.stride as u32;
        ur.zs[1] = input.plane_len() as u32;
        let ub = passes.uniform("lens flare restore uniform", &ur)?;
        passes.dispatch(
            KernelId::LensFlareRestore,
            "lens flare restore",
            &[
                (0, ub.binding()),
                (2, input.buf.binding()),
                (8, res.binding()),
            ],
            (
                (region.w as u32).div_ceil(GROUP),
                (region.h as u32).div_ceil(GROUP),
                1,
            ),
        )?;
    }
    for (li, grid) in grids.iter().enumerate() {
        if grid_set & (1 << li) == 0 {
            continue;
        }
        let mut un = u;
        un.lvl = [li as u32, 0, 0, 0];
        let ub = passes.uniform("lens flare nodes uniform", &un)?;
        passes.dispatch(
            KernelId::LensFlareNodes,
            "lens flare nodes",
            &[
                (0, ub.binding()),
                (5, scale.vectors.binding()),
                (6, scale.bins.binding()),
                (7, lens.buf.binding()),
                (8, nodes.binding()),
            ],
            (
                (grid[2] as u32).div_ceil(GROUP),
                (grid[3] as u32).div_ceil(GROUP),
                1,
            ),
        )?;
    }
    let ub = passes.uniform("lens flare uniform", &u)?;
    let res_binding = restored
        .as_ref()
        .map_or_else(|| scale.vectors.binding(), |r| r.binding());
    passes.dispatch(
        KernelId::LensFlare,
        "lens flare",
        &[
            (0, ub.binding()),
            (1, res_binding),
            (2, z.buf.binding()),
            (3, dst.buf.binding()),
            (4, planes.buf.binding()),
            (5, scale.vectors.binding()),
            (6, scale.bins.binding()),
            (7, lens.buf.binding()),
            (8, nodes.binding()),
        ],
        (
            (out.w as u32).div_ceil(GROUP),
            (out.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    passes.hold(Arc::clone(&planes.buf));
    passes.hold(Arc::clone(&lens.buf));
    passes.hold(Arc::clone(&scale.vectors));
    passes.hold(Arc::clone(&scale.bins));
    Ok(dst)
}

/// L3 on the host's behalf: uploads everything, runs the pass over `out` and reads the planes back
/// (`3 * out.w * out.h` samples). `input` (three planes, `input_size`, first pixel at `input_at`) is the
/// picture before L1 and `z` the output of L2 over `out` exactly. The GPU counterpart of
/// `ym_lens::apply_flares`; for tests and the first wiring.
///
/// # Errors
///
/// As [`encode_flare`], plus the readback's.
#[allow(clippy::too_many_arguments)]
pub fn flare(
    ctx: &GpuContext,
    planes: &FlarePlaneData,
    lens: &FlareLensData,
    scale: &FlareScaleData,
    mix: FlareMix,
    input: &[f32],
    input_size: (usize, usize),
    input_at: (usize, usize),
    z: &[f32],
    out: Rect,
) -> Result<Vec<f32>, GpuError> {
    let place = Placement {
        x0: input_at.0,
        y0: input_at.1,
        frame_width: scale.width,
        frame_height: scale.height,
    };
    let src = GpuImage::upload(
        ctx,
        "lens flare input",
        input_size.0,
        input_size.1,
        3,
        input,
        place,
    )?;
    let zimg = GpuImage::upload(
        ctx,
        "lens flare z",
        out.w,
        out.h,
        3,
        z,
        Placement {
            x0: out.x,
            y0: out.y,
            frame_width: scale.width,
            frame_height: scale.height,
        },
    )?;
    let planes = FlarePlaneBuffer::upload(ctx, planes)?;
    let lens = FlareLensBuffer::upload(ctx, lens)?;
    let scale = FlareScaleBuffers::upload(ctx, scale)?;
    let mut passes = PassList::new(ctx, "lens flare");
    let img = encode_flare(
        &mut passes,
        &planes,
        &lens,
        &scale,
        mix,
        &src,
        input_at,
        &zimg,
        (out.x, out.y),
        out,
    )?;
    let read = img.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    img.take(&done, read)
}

fn constants() -> String {
    format!(
        "const FL_SPIKE: u32 = {SPIKE_SIZE}u;\n\
         const FL_TAB_RIM_M: u32 = {}u;\n\
         const FL_TAB_APERTURE: u32 = {}u;\n\
         const FL_TAB_NODES: u32 = {FLARE_TAB_HEADER}u;\n\
         const FL_NODES: u32 = {NORMAL_TAIL_NODES}u;\n\
         const FL_STREAK_FLAT: f32 = {:?};\n",
        RIM_SAMPLES / 4,
        2 * RIM_SAMPLES / 4,
        0.8f32,
    )
}

/// The WGSL of a flare kernel.
pub(super) fn body(id: KernelId) -> String {
    let main = match id {
        KernelId::LensFlareRestore => "lfl_restore_main",
        KernelId::LensFlareNodes => "lfl_nodes_main",
        _ => "lfl_main_main",
    };
    let mut s = constants();
    s.push_str(include_str!("../shaders/lens_flare.wgsl"));
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
            KernelId::LensFlareRestore,
            KernelId::LensFlareNodes,
            KernelId::LensFlare,
        ] {
            let (fields, span) = uniform_layout(id, "pu");
            let want = [
                ("zero", offset_of!(FlareU, zero)),
                ("frame", offset_of!(FlareU, frame)),
                ("mix", offset_of!(FlareU, mix)),
                ("ints", offset_of!(FlareU, ints)),
                ("outr", offset_of!(FlareU, outr)),
                ("resr", offset_of!(FlareU, resr)),
                ("zr", offset_of!(FlareU, zr)),
                ("zs", offset_of!(FlareU, zs)),
                ("offs", offset_of!(FlareU, offs)),
                ("bins", offset_of!(FlareU, bins)),
                ("tail", offset_of!(FlareU, tail)),
                ("clip", offset_of!(FlareU, clip)),
                ("ramp", offset_of!(FlareU, ramp)),
                ("lg", offset_of!(FlareU, lg)),
                ("lgo", offset_of!(FlareU, lgo)),
                ("lvl", offset_of!(FlareU, lvl)),
            ];
            assert_eq!(fields.len(), want.len(), "{id:?}");
            for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
                assert_eq!(name, want_name);
                assert_eq!(*offset as usize, want_offset, "{id:?}: offset of {name}");
            }
            assert_eq!(span as usize, size_of::<FlareU>(), "{id:?}");
        }
    }

    #[test]
    fn the_table_layout_matches_the_planner() {
        // The rim's samples and second derivatives, the aperture header and its normals.
        assert_eq!(
            FLARE_TAB_HEADER,
            2 * RIM_SAMPLES / 4 + 1 + ym_lens::device::MAX_APERTURE_NORMALS / 2
        );
    }
}

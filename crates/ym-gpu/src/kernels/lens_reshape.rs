//! L1, the lens's bokeh reshaping, on the device (`docs/21` section 4, L1; `docs/17-gpu.md` section 11).
//!
//! The CPU reference is `ym_lens::l1` (reached through [`ym_lens::reshape`]): the photo's blur map is
//! brought to the tile's pixels by a joint bilateral upsampling guided by the tile's own colours and zeroed
//! where the picture shows an edge sharper than its blur; the bright balls are taken out of the picture; two
//! gathers, across the anamorphic axis and along the tangent, with lengths set by the blur; and the balls drawn
//! back as ellipses with the energy they had. This module is the same computation in `f32`.
//!
//! # Passes
//!
//! Twelve dispatches over the regions of one tile ([`ym_lens::device::BokehTile`]; `lens_reshape.wgsl` lists them):
//! the cells' colour guide, the blur map over `rb`, the sharp places (four passes over fields grown by 6, 4 and 2
//! pixels), the balls' protection and floor over `rb`, the feather's counts and the ramp over `r1`, the balls
//! taken out over `r0`, the vertical gather over `r1`, and the tangential gather with the balls drawn back over
//! the output.
//!
//! # Data
//!
//! Two kinds, by how long they live ([`ym_lens::device`]): the photo's [`BokehPlaneBuffer`] (the blur map's levels
//! and the table of blurred cells, uploaded once with the planes) and the scale's [`BokehScaleBuffers`] (the lens's
//! profiles, the map level's cell geometry, the balls' shapes and the bins that cull them; they depend on
//! Anamorphic and Edges, not on Flares). The numbers of the pass are [`ym_lens::device::BokehPass`].
//!
//! # Positions
//!
//! D4 and D5 of `docs/17-gpu.md`: a pixel is its absolute column and row; a ball's centre and a cell's centre are
//! an integer and a fraction, so the offsets that enter a distance are small; a tap is the pixel plus an offset of
//! at most the line's length. The balls a pixel meets are those of its bin of the absolute pixel grid, in list
//! order, so a tile and the whole frame draw the same balls in the same order. The vertical gather's exact mean is
//! a direct sum over the line's at most `2 l + 3` pixels, so it does not depend on where a tile starts.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ym_lens::PixelRect;
use ym_lens::consts::{
    ACCEPT_COLOUR, ACCEPT_CONFIDENCE, B_CAP, B_MIN, BALL_RIM_REACH_PX, MAX_STRETCH_TAPS, MIN_LINE_PX,
    SHARP_DILATE, SHARP_RATIO, SHARP_TEST_FROM_PX, STRAND_MARGIN_NOISE, STRAND_VALLEY,
    SUBJECT_SHARE,
};
use ym_lens::device::{
    BOKEH_BIN_PX, BOKEH_HEADER, BOKEH_SHAPE_VECTORS, BokehPass, BokehPlaneData, BokehScaleData,
    BokehTile,
};

use super::shader::KernelId;
use super::util::{GROUP, storage_buffer};
use crate::cpu::Placement;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// The uniform block of `lens_reshape.wgsl` (`ReshapeU`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReshapeU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    /// Frame width, height, half-diagonal in pixels, noise of the square-root luminance.
    frame: [f32; 4],
    /// Stretch per unit of blur, Edges, the guide's width squared, the upsampling's support.
    par: [f32; 4],
    /// The level's cell size in pixels along x and y, the sharp test's gate, 0.
    par2: [f32; 4],
    /// Frame width, height, 1 when the gather lines run along y, the feather.
    ints: [i32; 4],
    /// 1 with the vertical gather, 1 with the tangential gather, the balls, 0.
    mode: [i32; 4],
    /// The level's first cell, its columns, its rows, level 0's columns.
    lvl: [i32; 4],
    /// Level 0's rows, the first cell column read, the first cell row read, the cell columns read.
    map: [i32; 4],
    /// The cell rows read, one past the last cell column read, one past the last cell row read, 0.
    cellr: [i32; 4],
    /// The table's offsets (in vectors) of the columns' cells, the rows' cells, the balls; the bins' lists.
    offs: [u32; 4],
    /// Bins across, bins down, the bin side in pixels, 0.
    bins: [u32; 4],
    /// The input: frame column and row of its first pixel, width, height.
    src: [i32; 4],
    /// The input: row stride, plane length.
    srcs: [u32; 4],
    /// The output rectangle.
    wnt: [i32; 4],
    /// The tangential gather's input.
    r1: [i32; 4],
    /// The vertical gather's input, and the picture the balls are taken out of.
    r0: [i32; 4],
    /// The blur map's region.
    rb: [i32; 4],
    /// `rb` grown by 6, 4 and 2.
    e6: [i32; 4],
    /// See [`ReshapeU::e6`].
    e4: [i32; 4],
    /// See [`ReshapeU::e6`].
    e2: [i32; 4],
    /// The counts' first row and number of rows, 0, 0.
    cnt: [i32; 4],
    /// Scratch offsets (`f32` elements): the square roots, the energies, the mean's energies, the flags.
    scr: [u32; 4],
    /// Output row stride, plane length.
    dst: [u32; 4],
}

/// A photo's blur map on the device: the levels' cells and the table of blurred cells.
#[derive(Clone)]
pub struct BokehPlaneBuffer {
    cells: Arc<PooledBuffer>,
    sat: Arc<PooledBuffer>,
}

impl std::fmt::Debug for BokehPlaneBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BokehPlaneBuffer")
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl BokehPlaneBuffer {
    /// Uploads a photo's blur map.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a table of the wrong length; the pool's errors.
    pub fn upload(ctx: &GpuContext, data: &BokehPlaneData) -> Result<Self, GpuError> {
        let (w, h) = data.size;
        let total: usize = data.levels.iter().map(|(lw, lh, _)| lw * lh).sum();
        if data.cells.len() != total || data.nonzero.len() != (w + 1) * (h + 1) {
            return Err(GpuError::InvalidInput(format!(
                "lens bokeh map: {} cells for levels of {total}, {} table entries for a {w} x {h} map",
                data.cells.len(),
                data.nonzero.len()
            )));
        }
        Ok(BokehPlaneBuffer {
            cells: Arc::new(storage_buffer(
                ctx,
                "lens bokeh map",
                bytemuck::cast_slice(&data.cells),
            )?),
            sat: Arc::new(storage_buffer(
                ctx,
                "lens bokeh table",
                bytemuck::cast_slice(&data.nonzero),
            )?),
        })
    }

    /// Bytes on the device.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.cells.size() + self.sat.size()
    }
}

/// One scale's profiles, cell geometry and balls on the device.
#[derive(Clone)]
pub struct BokehScaleBuffers {
    tab: Arc<PooledBuffer>,
    bins: Arc<PooledBuffer>,
    pass: BokehPass,
    x_offset: usize,
    y_offset: usize,
    shape_offset: usize,
    shapes: usize,
    bins_x: usize,
    bins_y: usize,
    list_offset: usize,
}

impl std::fmt::Debug for BokehScaleBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BokehScaleBuffers")
            .field("frame", &(self.pass.width, self.pass.height))
            .field("balls", &self.shapes)
            .finish()
    }
}

impl BokehScaleBuffers {
    /// Uploads a scale's data (`LensPlan::bokeh_scale_data`).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for data that do not fit their own layout; the pool's errors.
    pub fn upload(ctx: &GpuContext, data: &BokehScaleData) -> Result<Self, GpuError> {
        let nb = data.bins_x * data.bins_y;
        let (lw, lh) = data.pass.level_size;
        if data.vectors.len() != data.shape_offset + data.shapes * BOKEH_SHAPE_VECTORS
            || data.x_offset != BOKEH_HEADER
            || data.y_offset != data.x_offset + lw
            || data.shape_offset != data.y_offset + lh
            || data.bins.len() < nb + 1
            || data.bins_x != data.pass.width.div_ceil(BOKEH_BIN_PX)
        {
            return Err(GpuError::InvalidInput(
                "lens bokeh scale data: the layout does not match its sizes".into(),
            ));
        }
        Ok(BokehScaleBuffers {
            tab: Arc::new(storage_buffer(
                ctx,
                "lens bokeh scale",
                bytemuck::cast_slice(&data.vectors),
            )?),
            bins: Arc::new(storage_buffer(
                ctx,
                "lens bokeh bins",
                bytemuck::cast_slice(&data.bins),
            )?),
            pass: data.pass,
            x_offset: data.x_offset,
            y_offset: data.y_offset,
            shape_offset: data.shape_offset,
            shapes: data.shapes,
            bins_x: data.bins_x,
            bins_y: data.bins_y,
            list_offset: nb + 1,
        })
    }

    /// The numbers of the pass these buffers belong to.
    #[must_use]
    pub fn pass(&self) -> &BokehPass {
        &self.pass
    }

    /// Pixels of context an output rectangle needs around it.
    #[must_use]
    pub fn halo(&self) -> usize {
        self.pass.halo()
    }

    /// Bytes on the device.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.tab.size() + self.bins.size()
    }
}

fn pixel_rect(r: Rect) -> PixelRect {
    PixelRect::new(r.x, r.y, r.w, r.h)
}

fn ints(r: PixelRect) -> [i32; 4] {
    [r.x as i32, r.y as i32, r.w as i32, r.h as i32]
}

/// Where the buffers of a tile's passes lie, from its rectangles.
struct Layout {
    tile: BokehTile,
    e6: PixelRect,
    e4: PixelRect,
    e2: PixelRect,
    /// First row and number of rows of the feather's counts (columns are `r1`'s).
    cnt_rows: (usize, usize),
    /// Offsets in the scratch buffer of the square roots, the two energies and the flags (`f32` elements).
    scr_offsets: [usize; 4],
    /// Elements of the scratch buffer.
    scr_len: usize,
}

fn layout(pass: &BokehPass, want: Rect) -> Layout {
    let (fw, fh) = (pass.width, pass.height);
    let tile = pass.tile(pixel_rect(want));
    // The fields of the sharp-edge test: the luminance root, the gradient energies and the flags, each reading
    // the one before within a pixel, two pixels or (the flags' dilation) `SHARP_DILATE`.
    let (e6, e4, e2) = (
        tile.rb.grown(SHARP_DILATE + 4, fw, fh),
        tile.rb.grown(SHARP_DILATE + 2, fw, fh),
        tile.rb.grown(SHARP_DILATE, fw, fh),
    );
    let f = pass.feather;
    let first = tile.r1.y.saturating_sub(f);
    let last = (tile.r1.y1() + f).min(fh);
    let area = |r: PixelRect| r.w * r.h;
    let a = [
        0,
        area(e6),
        area(e6) + area(e4),
        area(e6) + 2 * area(e4),
    ];
    let flags_chain = a[3] + area(e2);
    let taken_out = 3 * area(tile.r0);
    Layout {
        tile,
        e6,
        e4,
        e2,
        cnt_rows: (first, last.saturating_sub(first)),
        scr_offsets: a,
        scr_len: flags_chain.max(taken_out),
    }
}

/// The largest storage binding (bytes) the pass needs for an output of `want`: the input window, the scratch
/// buffer or one of the tile's planes.
#[must_use]
pub fn max_binding_bytes(want: Rect, pass: &BokehPass) -> u64 {
    let l = layout(pass, want);
    let t = &l.tile;
    let plane = |r: PixelRect| (r.w * r.h) as u64 * 4;
    GpuImage::buffer_bytes(t.need.w, t.need.h, 3)
        .max(l.scr_len as u64 * 4)
        .max(plane(t.rb))
        .max(3 * plane(t.r1))
        .max(GpuImage::buffer_bytes(want.w, want.h, 3))
}

/// Records L1 over `want` (frame coordinates at the scale of `scale`): device in, device out.
///
/// `src` is a three-plane image whose first pixel is the frame pixel `at`; it must cover `want` grown by
/// [`BokehScaleBuffers::halo`] and clamped to the frame (the image's own `place` is not read). The result is
/// `want.w x want.h`, placed at `(want.x, want.y)` in the frame; it is what the field optics read.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a source that is not three planes or does not cover the halo, or an empty
/// output or one outside the frame; the pool's errors; shader errors.
pub fn encode_reshape(
    passes: &mut PassList<'_>,
    planes: &BokehPlaneBuffer,
    scale: &BokehScaleBuffers,
    src: &GpuImage,
    at: (usize, usize),
    want: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    let pass = &scale.pass;
    let (fw, fh) = (pass.width, pass.height);
    if src.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "lens reshape: the source has {} planes, not 3",
            src.planes
        )));
    }
    if want.is_empty() || want.x1() > fw || want.y1() > fh {
        return Err(GpuError::InvalidInput(format!(
            "lens reshape: the output {want:?} is empty or outside the {fw} x {fh} frame"
        )));
    }
    let l = layout(pass, want);
    let t = l.tile;
    let have = PixelRect::new(at.0, at.1, src.width, src.height);
    if !have.contains(&t.need) {
        return Err(GpuError::InvalidInput(format!(
            "lens reshape: the source {have} does not cover the output {want:?} grown by {} px",
            pass.halo()
        )));
    }
    let dst = GpuImage::alloc(
        ctx,
        "lens reshaped",
        want.w,
        want.h,
        3,
        Placement {
            x0: want.x,
            y0: want.y,
            frame_width: fw,
            frame_height: fh,
        },
    )?;
    let cells = pass.cells_of(t.rb);
    let (ncx, ncy) = (
        cells.x_hi.saturating_sub(cells.x_lo),
        cells.y_hi.saturating_sub(cells.y_lo),
    );
    let guide = passes.scratch("lens bokeh guide", (ncx * ncy).max(1) as u64 * 16)?;
    let bt = passes.scratch("lens bokeh blur", (t.rb.w * t.rb.h) as u64 * 4)?;
    let info = passes.scratch("lens bokeh flags", (t.rb.w * t.rb.h) as u64 * 4)?;
    let scr = passes.scratch("lens bokeh scratch", l.scr_len as u64 * 4)?;
    let o1 = passes.scratch("lens bokeh vertical", 3 * (t.r1.w * t.r1.h) as u64 * 4)?;
    let ramp = passes.scratch("lens bokeh ramp", (t.r1.w * t.r1.h) as u64 * 4)?;
    let counts = passes.scratch("lens bokeh counts", (t.r1.w * l.cnt_rows.1).max(1) as u64 * 4)?;

    let mut u = ReshapeU::zeroed();
    u.frame = [
        fw as f32,
        fh as f32,
        pass.half_diag as f32,
        pass.noise as f32,
    ];
    u.par = [
        pass.stretch as f32,
        pass.edges as f32,
        (pass.sigma_guide * pass.sigma_guide) as f32,
        pass.support as f32,
    ];
    // The sharp test's gate: SHARP_GATE noise variances over a 5 x 5 window.
    let gate = (ym_lens::consts::SHARP_GATE * 25.0 * pass.noise * pass.noise) as f32;
    u.par2 = [pass.cell.0 as f32, pass.cell.1 as f32, gate, 0.0];
    u.ints = [
        fw as i32,
        fh as i32,
        i32::from(pass.lines_along_y),
        pass.feather as i32,
    ];
    u.mode = [
        i32::from(pass.vertical),
        i32::from(pass.tangential),
        scale.shapes as i32,
        0,
    ];
    u.lvl = [
        pass.level_offset as i32,
        pass.level_size.0 as i32,
        pass.level_size.1 as i32,
        pass.map.0 as i32,
    ];
    u.map = [
        pass.map.1 as i32,
        cells.x_lo as i32,
        cells.y_lo as i32,
        ncx as i32,
    ];
    u.cellr = [ncy as i32, cells.x_hi as i32, cells.y_hi as i32, 0];
    u.offs = [
        scale.x_offset as u32,
        scale.y_offset as u32,
        scale.shape_offset as u32,
        scale.list_offset as u32,
    ];
    u.bins = [scale.bins_x as u32, scale.bins_y as u32, BOKEH_BIN_PX as u32, 0];
    u.src = [
        at.0 as i32,
        at.1 as i32,
        src.width as i32,
        src.height as i32,
    ];
    u.srcs = [src.stride as u32, src.plane_len() as u32, 0, 0];
    u.wnt = ints(pixel_rect(want));
    u.r1 = ints(t.r1);
    u.r0 = ints(t.r0);
    u.rb = ints(t.rb);
    u.e6 = ints(l.e6);
    u.e4 = ints(l.e4);
    u.e2 = ints(l.e2);
    u.cnt = [l.cnt_rows.0 as i32, l.cnt_rows.1 as i32, 0, 0];
    u.scr = [
        l.scr_offsets[0] as u32,
        l.scr_offsets[1] as u32,
        l.scr_offsets[2] as u32,
        l.scr_offsets[3] as u32,
    ];
    u.dst = [dst.stride as u32, dst.plane_len() as u32, 0, 0];

    let groups = |w: usize, h: usize| ((w as u32).div_ceil(GROUP), (h as u32).div_ceil(GROUP), 1);
    let run = |passes: &mut PassList<'_>,
                   id: KernelId,
                   label: &str,
                   entries: &[(u32, wgpu::BindingResource<'_>)],
                   size: (usize, usize)|
     -> Result<(), GpuError> {
        let ub = passes.uniform(&format!("{label} uniform"), &u)?;
        let mut all = vec![(0, ub.binding())];
        all.extend(entries.iter().cloned());
        passes.dispatch(id, label, &all, groups(size.0, size.1))
    };

    // The cells' colour guide, then the blur map.
    run(
        passes,
        KernelId::LensReshapeGuide,
        "lens bokeh guide",
        &[
            (1, src.buf.binding()),
            (8, scale.tab.binding()),
            (12, guide.binding()),
        ],
        (ncx, ncy),
    )?;
    run(
        passes,
        KernelId::LensReshapeBlur,
        "lens bokeh blur",
        &[
            (1, src.buf.binding()),
            (2, bt.binding()),
            (8, scale.tab.binding()),
            (10, planes.cells.binding()),
            (11, planes.sat.binding()),
            (12, guide.binding()),
        ],
        (t.rb.w, t.rb.h),
    )?;
    // The sharp places.
    run(
        passes,
        KernelId::LensReshapeLumRoot,
        "lens bokeh luminance",
        &[(1, src.buf.binding()), (4, scr.binding())],
        (l.e6.w, l.e6.h),
    )?;
    run(
        passes,
        KernelId::LensReshapeGrad,
        "lens bokeh gradient",
        &[(4, scr.binding())],
        (l.e4.w, l.e4.h),
    )?;
    run(
        passes,
        KernelId::LensReshapeFlags,
        "lens bokeh flags",
        &[(4, scr.binding())],
        (l.e2.w, l.e2.h),
    )?;
    run(
        passes,
        KernelId::LensReshapeDilate,
        "lens bokeh dilate",
        &[(3, info.binding()), (4, scr.binding())],
        (t.rb.w, t.rb.h),
    )?;
    // The balls' protection and floor, the ramp.
    run(
        passes,
        KernelId::LensReshapeProtect,
        "lens bokeh protect",
        &[
            (1, src.buf.binding()),
            (2, bt.binding()),
            (3, info.binding()),
            (8, scale.tab.binding()),
            (9, scale.bins.binding()),
            (10, planes.cells.binding()),
            (12, guide.binding()),
        ],
        (t.rb.w, t.rb.h),
    )?;
    if pass.feather > 0 {
        run(
            passes,
            KernelId::LensReshapeCover,
            "lens bokeh cover",
            &[(2, bt.binding()), (13, counts.binding())],
            (t.r1.w, l.cnt_rows.1),
        )?;
    }
    run(
        passes,
        KernelId::LensReshapeRamp,
        "lens bokeh ramp",
        &[
            (3, info.binding()),
            (7, ramp.binding()),
            (13, counts.binding()),
        ],
        (t.r1.w, t.r1.h),
    )?;
    // The balls taken out, the two gathers, the balls drawn back.
    run(
        passes,
        KernelId::LensReshapeTakeOut,
        "lens bokeh take out",
        &[
            (1, src.buf.binding()),
            (2, bt.binding()),
            (4, scr.binding()),
            (8, scale.tab.binding()),
            (9, scale.bins.binding()),
        ],
        (t.r0.w, t.r0.h),
    )?;
    run(
        passes,
        KernelId::LensReshapeVGather,
        "lens bokeh vertical gather",
        &[
            (2, bt.binding()),
            (4, scr.binding()),
            (5, o1.binding()),
            (7, ramp.binding()),
        ],
        (t.r1.w, t.r1.h),
    )?;
    run(
        passes,
        KernelId::LensReshapeFinal,
        "lens bokeh tangential gather",
        &[
            (2, bt.binding()),
            (3, info.binding()),
            (5, o1.binding()),
            (6, dst.buf.binding()),
            (7, ramp.binding()),
            (8, scale.tab.binding()),
            (9, scale.bins.binding()),
        ],
        (want.w, want.h),
    )?;
    passes.hold(Arc::clone(&planes.cells));
    passes.hold(Arc::clone(&planes.sat));
    passes.hold(Arc::clone(&scale.tab));
    passes.hold(Arc::clone(&scale.bins));
    Ok(dst)
}

/// L1 on the host's behalf: uploads everything, runs the pass over `want` and reads the planes back
/// (`3 * want.w * want.h` samples). `input` (three planes, `size`, first pixel at `at`) is the picture before
/// L1. The GPU counterpart of `ym_lens::reshape`; for tests and the first wiring.
///
/// # Errors
///
/// As [`encode_reshape`], plus the readback's.
pub fn reshape(
    ctx: &GpuContext,
    planes: &BokehPlaneData,
    scale: &BokehScaleData,
    input: &[f32],
    size: (usize, usize),
    at: (usize, usize),
    want: Rect,
) -> Result<Vec<f32>, GpuError> {
    let src = GpuImage::upload(
        ctx,
        "lens reshape input",
        size.0,
        size.1,
        3,
        input,
        Placement {
            x0: at.0,
            y0: at.1,
            frame_width: scale.pass.width,
            frame_height: scale.pass.height,
        },
    )?;
    let planes = BokehPlaneBuffer::upload(ctx, planes)?;
    let scale = BokehScaleBuffers::upload(ctx, scale)?;
    let mut passes = PassList::new(ctx, "lens reshape");
    let img = encode_reshape(&mut passes, &planes, &scale, &src, at, want)?;
    let read = img.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    img.take(&done, read)
}

fn constants() -> String {
    let f = |x: f64| format!("{:?}", x as f32);
    format!(
        "const LR_B_MIN: f32 = {};\n\
         const LR_B_CAP: f32 = {};\n\
         const LR_MIN_LINE: f32 = {};\n\
         const LR_ACCEPT2: f32 = {};\n\
         const LR_ACCEPT_CONF: f32 = {};\n\
         const LR_SUBJECT_SHARE: f32 = {};\n\
         const LR_SHARP_FROM: f32 = {};\n\
         const LR_SHARP_RATIO: f32 = {};\n\
         const LR_STRAND_VALLEY: f32 = {};\n\
         const LR_STRAND_MARGIN: f32 = {};\n\
         const LR_RIM_REACH: f32 = {};\n\
         const LR_DILATE: i32 = {SHARP_DILATE};\n\
         const LR_HALF_TAPS: i32 = {};\n\
         const LR_SHAPE_VECTORS: u32 = {BOKEH_SHAPE_VECTORS}u;\n",
        f(B_MIN),
        f(B_CAP),
        f(MIN_LINE_PX),
        f(ACCEPT_COLOUR * ACCEPT_COLOUR),
        f(ACCEPT_CONFIDENCE),
        f(SUBJECT_SHARE),
        f(SHARP_TEST_FROM_PX),
        f(SHARP_RATIO),
        f(STRAND_VALLEY),
        f(STRAND_MARGIN_NOISE),
        f(BALL_RIM_REACH_PX),
        (MAX_STRETCH_TAPS - 1) / 2,
    )
}

/// The WGSL of a reshaping kernel.
pub(super) fn body(id: KernelId) -> String {
    let main = match id {
        KernelId::LensReshapeGuide => "lr_guide_main",
        KernelId::LensReshapeBlur => "lr_blur_main",
        KernelId::LensReshapeLumRoot => "lr_lumroot_main",
        KernelId::LensReshapeGrad => "lr_grad_main",
        KernelId::LensReshapeFlags => "lr_flags_main",
        KernelId::LensReshapeDilate => "lr_dilate_main",
        KernelId::LensReshapeProtect => "lr_protect_main",
        KernelId::LensReshapeCover => "lr_cover_main",
        KernelId::LensReshapeRamp => "lr_ramp_main",
        KernelId::LensReshapeTakeOut => "lr_takeout_main",
        KernelId::LensReshapeVGather => "lr_vgather_main",
        _ => "lr_final_main",
    };
    let mut s = constants();
    s.push_str(include_str!("../shaders/lens_reshape.wgsl"));
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
            KernelId::LensReshapeGuide,
            KernelId::LensReshapeBlur,
            KernelId::LensReshapeLumRoot,
            KernelId::LensReshapeGrad,
            KernelId::LensReshapeFlags,
            KernelId::LensReshapeDilate,
            KernelId::LensReshapeProtect,
            KernelId::LensReshapeCover,
            KernelId::LensReshapeRamp,
            KernelId::LensReshapeTakeOut,
            KernelId::LensReshapeVGather,
            KernelId::LensReshapeFinal,
        ] {
            let (fields, span) = uniform_layout(id, "pu");
            let want = [
                ("zero", offset_of!(ReshapeU, zero)),
                ("frame", offset_of!(ReshapeU, frame)),
                ("par", offset_of!(ReshapeU, par)),
                ("par2", offset_of!(ReshapeU, par2)),
                ("ints", offset_of!(ReshapeU, ints)),
                ("mode", offset_of!(ReshapeU, mode)),
                ("lvl", offset_of!(ReshapeU, lvl)),
                ("map", offset_of!(ReshapeU, map)),
                ("cellr", offset_of!(ReshapeU, cellr)),
                ("offs", offset_of!(ReshapeU, offs)),
                ("bins", offset_of!(ReshapeU, bins)),
                ("src", offset_of!(ReshapeU, src)),
                ("srcs", offset_of!(ReshapeU, srcs)),
                ("wnt", offset_of!(ReshapeU, wnt)),
                ("r1", offset_of!(ReshapeU, r1)),
                ("r0", offset_of!(ReshapeU, r0)),
                ("rb", offset_of!(ReshapeU, rb)),
                ("e6", offset_of!(ReshapeU, e6)),
                ("e4", offset_of!(ReshapeU, e4)),
                ("e2", offset_of!(ReshapeU, e2)),
                ("cnt", offset_of!(ReshapeU, cnt)),
                ("scr", offset_of!(ReshapeU, scr)),
                ("dst", offset_of!(ReshapeU, dst)),
            ];
            assert_eq!(fields.len(), want.len(), "{id:?}");
            for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
                assert_eq!(name, want_name);
                assert_eq!(*offset as usize, want_offset, "{id:?}: offset of {name}");
            }
            assert_eq!(span as usize, size_of::<ReshapeU>(), "{id:?}");
        }
    }

    #[test]
    fn the_scratch_holds_the_flags_chain_and_then_the_taken_out_picture() {
        let pass = BokehPass {
            width: 3000,
            height: 2000,
            half_diag: 1803.0,
            noise: 0.01,
            stretch: 1.5,
            edges: 1.0,
            lines_along_y: true,
            vertical: true,
            tangential: true,
            v_halo: 80,
            t_halo: 40,
            feather: 20,
            guide_reach: 14,
            map: (750, 500),
            level: 0,
            level_size: (750, 500),
            level_offset: 0,
            cell: (4.0, 4.0),
            support: 6.4,
            sigma_guide: 0.06,
        };
        let l = layout(&pass, Rect::new(1000, 600, 1024, 800));
        // The chain's fields do not overlap, and the scratch holds the larger of the chain and the picture.
        let area = |r: PixelRect| r.w * r.h;
        assert_eq!(l.scr_offsets[1], area(l.e6));
        assert_eq!(l.scr_offsets[2], area(l.e6) + area(l.e4));
        assert_eq!(l.scr_offsets[3], area(l.e6) + 2 * area(l.e4));
        assert!(l.scr_len >= l.scr_offsets[3] + area(l.e2));
        assert!(l.scr_len >= 3 * area(l.tile.r0));
        assert!(l.tile.rb.contains(&l.tile.r0) && l.tile.r0.contains(&l.tile.r1));
        assert!(l.e6.contains(&l.e4) && l.e4.contains(&l.e2) && l.e2.contains(&l.tile.rb));
        // The counts' rows are inside the blur region.
        assert!(l.cnt_rows.0 >= l.tile.rb.y && l.cnt_rows.0 + l.cnt_rows.1 <= l.tile.rb.y1());
    }

}

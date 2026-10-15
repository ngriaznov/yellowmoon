//! The export tail on the device (`docs/gpu-plan.md` section 4.10, G-08): the rendered camera RGB of
//! an export goes through the geometry sampler, the colour chain C1 to C12, the Lanczos-3 resize,
//! output sharpening and C13, and comes back as encoded bands that the CPU writers pull. No
//! display-linear frame is ever held on the host: at 50 MP that frame is 600 MB.
//!
//! # What runs where
//!
//! The caller (`ym-core`) renders the raw chain tile by tile, as the CPU export does, and hands each
//! tile's camera RGB to [`ExportTail::add_tile`]. One device job per tile uploads it, samples it
//! through the photo's geometry ([`crate::kernels::sampler`]), runs the colour chain to C12
//! ([`crate::kernels::color::PreparedChain`], persistent tables) and writes the result into the
//! **slabs** of the rendered picture. After each row of tiles the caller says how many rows of the
//! picture are complete ([`ExportTail::advance`]) and the tail runs whatever it can, returning the
//! bands that are finished:
//!
//! ```text
//! tiles -> slabs (rendered picture, 4 planes, zero outside the photo)
//!       -> [horizontal resize, one job per slab] -> [vertical resize, one job per chunk of output rows]
//!       -> [output sharpening of a chunk of rows with its halo] -> C13 (+ dither) -> bytes -> host
//! ```
//!
//! The stages in brackets are only there when the export resizes or sharpens. Each stage reads rows
//! it needs from the rows the stage before it has made (a resize output needs a run of input rows,
//! sharpening needs the stage's halo above and below), so the tail is a pipeline of **row stores**:
//! a store holds the pieces (slabs or chunks) not yet consumed, a stage takes the rows it needs
//! (copying across piece borders when it must), and a piece is released as soon as no later row
//! needs it. The pieces are never larger than [`SLAB_BYTES`] (the design binding of the plan, N-1),
//! so at 61 MP the device holds a few slabs and one chunk of scratch, not the 700 MB picture.
//!
//! # Determinism
//!
//! The chunk boundaries are functions of the picture's size and the settings alone (never of timing,
//! the device or the tile order), and every kernel is per-pixel or tile-exact (D4, D5 of the plan):
//! the bytes do not depend on how the rows were cut, and `tests/gpu_tail.rs` checks it.
//!
//! # Cancel, errors
//!
//! Every job of a tail carries the render's cancel token and the request's priority and owner
//! ([`TailRoute`]); a render that is cancelled while a job is queued stops waiting at once. A device
//! error is the caller's to answer (the whole photo goes to the CPU, `docs/gpu-plan.md` 4.9): this
//! module never continues on the CPU. A buffer that does not fit (out of budget, too large) is tried
//! once more after the pool and the persistent colour tables were let go.

use std::collections::VecDeque;
use std::ops::{Deref, Range};
use std::sync::Arc;
use std::time::Duration;

use ym_color::display::ColorChain;
use ym_color::encode::OutputSpace;

use crate::cpu::output_sharpen::OutputSharpen;
use crate::cpu::sampler::SamplerGrid;
use crate::cpu::tiling::Rect;
use crate::cpu::{Placement, RgbImage};
use crate::device::{Gpu, GpuContext, GpuError};
use crate::image::GpuImage;
use crate::kernels::color::{PreparedChain, Stop, release_tables};
use crate::kernels::encode::{C13, Encoding, band_bytes, encode_c13, row_words};
use crate::kernels::output_sharpen::PreparedSharpen;
use crate::kernels::resize::{PreparedTaps, Taps, encode_resize_h, encode_resize_v};
use crate::kernels::sampler::encode_sample;
use crate::kernels::util::STORAGE_OUT;
use crate::queue::{CancelToken, JobSpec};
use crate::stream::PassList;
use crate::tiles::DESIGN_BINDING_BYTES;

/// The largest storage binding of one piece of the tail: the design binding of the plan (128 MiB,
/// `docs/gpu-plan.md` N-1), a constant and not the device's limit, so that the cut does not depend
/// on the machine.
pub const SLAB_BYTES: u64 = DESIGN_BINDING_BYTES;
/// Rows of a slab of the rendered picture at most: half a tile row of the export (1024 px), so that
/// at 61 MP (9504 px wide) a four-plane slab is 78 MB.
pub const SLAB_ROWS: usize = 512;
/// Pixels in one band that is read back when the export is not sharpened: about a million, like the
/// bands the CPU writers cut.
pub const BAND_PIXELS: usize = 1 << 20;
/// Pixels in one chunk of output rows of the vertical resize.
pub const RESIZE_CHUNK_PIXELS: usize = 1 << 20;
/// Pixels of the window of one output-sharpening chunk, halo included: the scratch of the stage is
/// 76 bytes a pixel, so this keeps it under 200 MB.
pub const SHARPEN_WINDOW_PIXELS: usize = 2_500_000;

/// How often a tail that waits for a device job looks at its cancel token.
const POLL: Duration = Duration::from_millis(5);

/// The depth the bands are encoded to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandFormat {
    /// 8 bits a channel, with the blue-noise dither when asked for (JPEG, 8-bit PNG and TIFF dither).
    Eight {
        /// Whether the codes are dithered; otherwise rounded.
        dither: bool,
    },
    /// 16 bits a channel, rounded, never dithered.
    Sixteen,
}

impl BandFormat {
    fn encoding(self) -> Encoding {
        match self {
            BandFormat::Eight { .. } => Encoding::Rgb8 { background: [0; 3] },
            BandFormat::Sixteen => Encoding::Rgb16 { bits: 16 },
        }
    }

    fn dither(self) -> bool {
        matches!(self, BandFormat::Eight { dither: true })
    }
}

/// Output sharpening of the picture at the size it is written.
#[derive(Clone, Debug)]
pub struct TailSharpen {
    /// The stage and its amount ([`OutputSharpen::screen`] with `with_amount_pct`).
    pub params: OutputSharpen,
    /// `s = long edge written / long edge rendered`
    /// ([`scale_for_edges`](crate::cpu::output_sharpen::scale_for_edges)).
    pub scale: f32,
}

/// What an export tail makes of the rendered picture.
#[derive(Clone)]
pub struct TailSpec {
    /// Size of the rendered picture (what the tiles cover).
    pub rendered: (usize, usize),
    /// Size of the written picture (equal to `rendered` when the export does not resize).
    pub out: (usize, usize),
    /// The colour chain, C1 to C12, for the export's output primaries.
    pub chain: Arc<ColorChain>,
    /// The geometry that maps the rendered picture's pixels to the camera RGB.
    pub grid: SamplerGrid,
    /// Output sharpening, when the export has any (a plan without a stage is dropped here).
    pub sharpen: Option<TailSharpen>,
    /// The transfer function of the written picture.
    pub space: OutputSpace,
    /// 8 or 16 bits.
    pub format: BandFormat,
}

/// The device, the queue priority and owner, and the cancel token every job of a tail carries.
#[derive(Clone)]
pub struct TailRoute<'a> {
    gpu: &'a Gpu,
    spec: JobSpec,
    token: CancelToken,
}

impl<'a> TailRoute<'a> {
    /// A route over `gpu`: jobs at `spec`, skipped (or stopped between steps) once `token` is set.
    #[must_use]
    pub fn new(gpu: &'a Gpu, spec: JobSpec, token: CancelToken) -> Self {
        Self { gpu, spec, token }
    }

    /// The device.
    #[must_use]
    pub fn gpu(&self) -> &'a Gpu {
        self.gpu
    }

    /// Runs `work` over `data` as one job of the route and waits for it.
    ///
    /// A device that is out of memory or refuses a buffer is given one more try after the pool let
    /// go of what it holds unused and the persistent colour tables were dropped. A render that is
    /// cancelled while its job is still queued stops waiting at once; the job is skipped when it is
    /// popped (its token is set).
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when the render was cancelled, the job's own error otherwise.
    pub fn run<D, R>(
        &self,
        data: D,
        work: fn(&GpuContext, &CancelToken, &D) -> Result<R, GpuError>,
    ) -> Result<R, GpuError>
    where
        D: Send + 'static,
        R: Send + 'static,
    {
        let handle =
            self.gpu
                .submit_with_token(self.spec, self.token.clone(), move |ctx, token| {
                    ctx.check_alive()?;
                    match work(ctx, token, &data) {
                        Err(GpuError::OutOfBudget { .. } | GpuError::BufferTooLarge { .. }) => {
                            ctx.pool().trim();
                            release_tables(ctx);
                            work(ctx, token, &data)
                        }
                        other => other,
                    }
                });
        loop {
            if let Some(done) = handle.wait_timeout(POLL) {
                return done?;
            }
            if self.token.is_cancelled() {
                return Err(GpuError::Cancelled);
            }
        }
    }
}

/// Encoded rows that came off the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Band {
    /// The first row of the written picture.
    pub y0: usize,
    /// Rows in the band.
    pub rows: usize,
    /// The codes, interleaved RGB, rows tight (no padding).
    pub data: BandData,
}

/// The codes of a [`Band`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BandData {
    /// `rows * width * 3` bytes.
    Eight(Vec<u8>),
    /// `rows * width * 3` samples.
    Sixteen(Vec<u16>),
}

// ---------------------------------------------------------------------------------------------
// Row stores
// ---------------------------------------------------------------------------------------------

/// Rows `y0..y0 + rows` of a picture, full width, on the device.
#[derive(Clone)]
struct Piece {
    y0: usize,
    rows: usize,
    image: Arc<GpuImage>,
}

impl Piece {
    fn end(&self) -> usize {
        self.y0 + self.rows
    }
}

/// The pieces of a picture that no stage has finished with yet, in order and without a gap.
#[derive(Default)]
struct Store {
    pieces: VecDeque<Piece>,
}

impl Store {
    /// The pieces that overlap `rows`.
    fn overlapping(&self, rows: &Range<usize>) -> Vec<Piece> {
        self.pieces
            .iter()
            .filter(|p| p.y0 < rows.end && p.end() > rows.start)
            .cloned()
            .collect()
    }

    /// Drops the pieces that end at or before `row`.
    fn release_below(&mut self, row: usize) {
        while self.pieces.front().is_some_and(|p| p.end() <= row) {
            self.pieces.pop_front();
        }
    }
}

/// The rows a stage reads: a piece itself when they lie in one, else a copy made for the stage.
enum Window {
    Piece(Arc<GpuImage>),
    Gathered(GpuImage),
}

impl Deref for Window {
    type Target = GpuImage;

    fn deref(&self) -> &GpuImage {
        match self {
            Window::Piece(p) => p,
            Window::Gathered(g) => g,
        }
    }
}

/// Rows `rows` of the picture the `pieces` hold, as an image and the rectangle of it that holds
/// them. When the rows span several pieces they are copied (three planes, one copy per plane and
/// piece) into an image of their own.
fn window(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    pieces: &[Piece],
    rows: Range<usize>,
) -> Result<(Window, Rect), GpuError> {
    let covers = pieces.first().is_some_and(|p| p.y0 <= rows.start)
        && pieces.last().is_some_and(|p| p.end() >= rows.end)
        && pieces.windows(2).all(|w| w[0].end() == w[1].y0);
    if rows.is_empty() || !covers {
        return Err(GpuError::InvalidInput(format!(
            "the export tail needs rows {rows:?}, the pieces it holds do not cover them"
        )));
    }
    let width = pieces[0].image.width;
    let n = rows.len();
    if let Some(p) = pieces
        .iter()
        .find(|p| p.y0 <= rows.start && p.end() >= rows.end)
    {
        return Ok((
            Window::Piece(Arc::clone(&p.image)),
            Rect::new(0, rows.start - p.y0, width, n),
        ));
    }
    let dst = GpuImage::alloc(ctx, "tail window", width, n, 3, Placement::whole(width, n))?;
    for p in pieces
        .iter()
        .filter(|p| p.y0 < rows.end && p.end() > rows.start)
    {
        let (a, b) = (rows.start.max(p.y0), rows.end.min(p.end()));
        if p.image.stride != dst.stride {
            return Err(GpuError::InvalidInput(
                "the pieces of the export tail have different strides".into(),
            ));
        }
        let len = ((b - a) * dst.stride * 4) as u64;
        for c in 0..3 {
            let from = ((c * p.image.plane_len() + (a - p.y0) * p.image.stride) * 4) as u64;
            let to = ((c * dst.plane_len() + (a - rows.start) * dst.stride) * 4) as u64;
            passes
                .encoder()
                .copy_buffer_to_buffer(&p.image.buf, from, &dst.buf, to, len);
        }
    }
    Ok((Window::Gathered(dst), Rect::new(0, 0, width, n)))
}

// ---------------------------------------------------------------------------------------------
// The jobs (run on the device thread)
// ---------------------------------------------------------------------------------------------

/// A slab of the rendered picture: four planes (red, green, blue and the letterbox mask), zeroed,
/// so that a tile that never arrives (outside the photo) is black.
fn slab_job(
    ctx: &GpuContext,
    token: &CancelToken,
    &(width, rows): &(usize, usize),
) -> Result<GpuImage, GpuError> {
    token.check()?;
    let image = GpuImage::alloc(
        ctx,
        "tail slab",
        width,
        rows,
        4,
        Placement::whole(width, rows),
    )?;
    let mut passes = PassList::new(ctx, "tail slab clear");
    passes.encoder().clear_buffer(&image.buf, 0, None);
    passes.submit()?.finish()?;
    Ok(image)
}

/// Where a tile's rows go: rows `tile_row0..tile_row0 + rows` of the coloured tile land in `image`
/// at `(dst_x, dst_y)`.
struct SlabTarget {
    image: Arc<GpuImage>,
    tile_row0: usize,
    rows: usize,
    dst_x: usize,
    dst_y: usize,
}

struct TileData {
    region: Arc<RgbImage>,
    origin: (usize, usize),
    grid: SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
    chain: Arc<ColorChain>,
    targets: Vec<SlabTarget>,
}

/// Uploads a tile's camera RGB, samples it, colours it to C12 and writes it into its slabs.
fn tile_job(ctx: &GpuContext, token: &CancelToken, d: &TileData) -> Result<(), GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "tail tile");
    let src = GpuImage::upload_rgb(ctx, "tail source", &d.region)?;
    let sampled = encode_sample(&mut passes, &src, d.origin, &d.grid, d.off, d.w, d.h)?;
    let prepared = PreparedChain::new(ctx, &d.chain)?;
    for t in &d.targets {
        prepared.encode_stage(
            &mut passes,
            &sampled,
            Rect::new(0, t.tile_row0, d.w, t.rows),
            Stop::C12,
            &t.image,
            (t.dst_x, t.dst_y),
        )?;
    }
    token.check()?;
    // `src` and `sampled` live until the work has finished.
    passes.submit()?.finish()?;
    Ok(())
}

struct PrepareData {
    h: Option<Taps>,
    v: Option<Taps>,
}

/// The horizontal and the vertical taps on the device (each only when the export resizes).
type TapPair = (Option<Arc<PreparedTaps>>, Option<Arc<PreparedTaps>>);

/// The device copies of the resize taps.
fn prepare_job(
    ctx: &GpuContext,
    token: &CancelToken,
    d: &PrepareData,
) -> Result<TapPair, GpuError> {
    token.check()?;
    let make = |t: &Option<Taps>| -> Result<Option<Arc<PreparedTaps>>, GpuError> {
        t.as_ref()
            .map(|t| PreparedTaps::new(ctx, t.clone()).map(Arc::new))
            .transpose()
    };
    Ok((make(&d.h)?, make(&d.v)?))
}

struct HData {
    taps: Arc<PreparedTaps>,
    slab: Piece,
}

/// The horizontal pass over one slab: the slab's rows, `out` columns wide, three planes.
fn h_job(ctx: &GpuContext, token: &CancelToken, d: &HData) -> Result<GpuImage, GpuError> {
    token.check()?;
    let ow = d.taps.taps().out_len();
    let rows = d.slab.rows;
    let mid = GpuImage::alloc(ctx, "tail mid", ow, rows, 3, Placement::whole(ow, rows))?;
    let mut passes = PassList::new(ctx, "tail resize h");
    encode_resize_h(&mut passes, &d.taps, &d.slab.image, 0..rows, &mid, 0)?;
    passes.submit()?.finish()?;
    Ok(mid)
}

struct VData {
    taps: Arc<PreparedTaps>,
    pieces: Vec<Piece>,
    out_rows: Range<usize>,
}

/// The vertical pass for a chunk of output rows, reading the input rows they need.
fn v_job(ctx: &GpuContext, token: &CancelToken, d: &VData) -> Result<GpuImage, GpuError> {
    token.check()?;
    let span = d.taps.taps().input_span(d.out_rows.clone());
    let mut passes = PassList::new(ctx, "tail resize v");
    let (win, rect) = window(ctx, &mut passes, &d.pieces, span.clone())?;
    let w = win.width;
    let n = d.out_rows.len();
    let out = GpuImage::alloc(ctx, "tail final", w, n, 3, Placement::whole(w, n))?;
    // Row 0 of the window image holds input row `span.start - rect.y`.
    encode_resize_v(
        &mut passes,
        &d.taps,
        &win,
        span.start - rect.y,
        d.out_rows.clone(),
        &out,
        0,
    )?;
    passes.submit()?.finish()?;
    Ok(out)
}

struct BandData2 {
    pieces: Vec<Piece>,
    /// The rows to encode (the core, when sharpening).
    rows: Range<usize>,
    /// The rows to read, the core grown by the halo and clipped to the picture (equal to `rows` when
    /// there is no sharpening).
    window: Range<usize>,
    sharpen: Option<TailSharpen>,
    format: BandFormat,
    space: OutputSpace,
}

/// Encodes a chunk of rows: (sharpen,) C13, and the readback. Returns the bytes with the rows
/// padded to whole groups of pixels ([`row_words`]).
fn band_job(ctx: &GpuContext, token: &CancelToken, d: &BandData2) -> Result<Vec<u8>, GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "tail band");
    let (win, rect) = window(ctx, &mut passes, &d.pieces, d.window.clone())?;
    let w = win.width;
    let n = d.rows.len();
    let encoding = d.format.encoding();
    let bytes = band_bytes(encoding, w, n);
    let dst = passes.keep(ctx.pool().acquire("tail band", bytes, STORAGE_OUT)?);
    let c13 = C13 {
        space: d.space,
        dither: d.format.dither(),
        origin: (0, d.rows.start),
    };
    // The rows to encode sit in `rect` of `win` (no sharpening) or are made from it (sharpening).
    let sharpened;
    if let Some(sh) = &d.sharpen {
        let prepared = PreparedSharpen::new(ctx, &sh.params, sh.scale)?;
        let core = Rect::new(rect.x, rect.y + (d.rows.start - d.window.start), w, n);
        sharpened = GpuImage::alloc(ctx, "tail sharpened", w, n, 3, Placement::whole(w, n))?;
        prepared.encode(&mut passes, &win, rect, core, &sharpened, (0, 0))?;
        encode_c13(
            &mut passes,
            &sharpened,
            Rect::new(0, 0, w, n),
            encoding,
            c13,
            &dst,
            0,
            0,
        )?;
    } else {
        encode_c13(&mut passes, &win, rect, encoding, c13, &dst, 0, 0)?;
    }
    let read = passes.read(&dst, 0, bytes)?;
    token.check()?;
    let mut done = passes.submit()?.finish()?;
    Ok(done.take(read))
}

/// Removes the row padding of a [`band_job`] result and widens 16-bit codes.
fn compact(format: BandFormat, bytes: &[u8], width: usize, rows: usize) -> BandData {
    let encoding = format.encoding();
    let row_len = row_words(encoding, width) * 4;
    match format {
        BandFormat::Eight { .. } => {
            let mut out = Vec::with_capacity(rows * width * 3);
            for r in 0..rows {
                out.extend_from_slice(&bytes[r * row_len..r * row_len + width * 3]);
            }
            BandData::Eight(out)
        }
        BandFormat::Sixteen => {
            let mut out = Vec::with_capacity(rows * width * 3);
            for r in 0..rows {
                out.extend(
                    bytes[r * row_len..r * row_len + width * 6]
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]])),
                );
            }
            BandData::Sixteen(out)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The tail
// ---------------------------------------------------------------------------------------------

/// The export tail of one photo (module notes). Feed it the tiles in order with
/// [`add_tile`](ExportTail::add_tile), tell it how many rows are complete with
/// [`advance`](ExportTail::advance), and collect the bands.
pub struct ExportTail<'a> {
    route: TailRoute<'a>,
    spec: TailSpec,
    /// Rows of a slab.
    slab_rows: usize,
    /// Rows of a band when the export is not sharpened.
    band_rows: usize,
    /// Output rows of a chunk of the vertical resize.
    v_rows: usize,
    /// Rows of the core of a sharpening chunk, and the stage's halo (`None`: no sharpening).
    sharpen: Option<(TailSharpen, usize, usize)>,
    /// The device copies of the resize taps, horizontal and vertical.
    taps: Option<(Arc<PreparedTaps>, Arc<PreparedTaps>)>,
    /// The slabs of the rendered picture that no stage has finished with.
    rendered: Store,
    /// Rows allocated in slabs (the rendered store's `end()` before any is released).
    allocated: usize,
    /// Rows of the rendered picture that are complete.
    rows_done: usize,
    /// The horizontally resized rows (when resizing) and how many are complete.
    mid: Store,
    mid_done: usize,
    /// The finished rows of the resized picture (when resizing).
    resized: Store,
    next_v: usize,
    /// The next row of the written picture to encode.
    next_band: usize,
    jobs: usize,
}

impl<'a> ExportTail<'a> {
    /// Plans the tail of `spec`: the slab and chunk sizes, the sharpening plan and the resize taps.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty picture, a picture whose slabs cannot fit one
    /// binding, or a sharpening the CPU plan rejects; the pool's errors for the taps.
    pub fn new(route: TailRoute<'a>, spec: TailSpec) -> Result<Self, GpuError> {
        let (fw, fh) = spec.rendered;
        let (ow, oh) = spec.out;
        if fw == 0 || fh == 0 || ow == 0 || oh == 0 {
            return Err(GpuError::InvalidInput(format!(
                "an export tail of {fw} x {fh} to {ow} x {oh} pixels is empty"
            )));
        }
        let row_bytes = 16 * fw.next_multiple_of(2) as u64;
        let fit = (SLAB_BYTES / row_bytes) as usize;
        if fit == 0 {
            return Err(GpuError::BufferTooLarge {
                requested: row_bytes,
                limit: SLAB_BYTES,
            });
        }
        let slab_rows = SLAB_ROWS.min(fit);
        let sharpen = match &spec.sharpen {
            Some(s) => {
                let plan = s
                    .params
                    .plan(s.scale)
                    .map_err(|e| GpuError::InvalidInput(e.to_string()))?;
                if plan.stages().is_empty() {
                    None
                } else {
                    let halo = plan.halo_px();
                    let core = (SHARPEN_WINDOW_PIXELS / ow)
                        .saturating_sub(2 * halo)
                        .clamp(16, 1024);
                    Some((s.clone(), core, halo))
                }
            }
            None => None,
        };
        let resizes = (ow, oh) != (fw, fh);
        let taps = if resizes {
            let data = PrepareData {
                h: Some(Taps::new(fw, ow)),
                v: Some(Taps::new(fh, oh)),
            };
            let (h, v) = route.run(data, prepare_job)?;
            h.zip(v)
        } else {
            None
        };
        Ok(Self {
            band_rows: (BAND_PIXELS / ow).clamp(8, 1024),
            v_rows: (RESIZE_CHUNK_PIXELS / ow).clamp(8, 512),
            route,
            spec,
            slab_rows,
            sharpen,
            taps,
            rendered: Store::default(),
            allocated: 0,
            rows_done: 0,
            mid: Store::default(),
            mid_done: 0,
            resized: Store::default(),
            next_v: 0,
            next_band: 0,
            jobs: 0,
        })
    }

    /// Device jobs submitted so far.
    #[must_use]
    pub fn jobs(&self) -> usize {
        self.jobs
    }

    /// Rows of the written picture that have been encoded.
    #[must_use]
    pub fn rows_encoded(&self) -> usize {
        self.next_band
    }

    fn run<D, R>(
        &mut self,
        data: D,
        work: fn(&GpuContext, &CancelToken, &D) -> Result<R, GpuError>,
    ) -> Result<R, GpuError>
    where
        D: Send + 'static,
        R: Send + 'static,
    {
        self.jobs += 1;
        self.route.run(data, work)
    }

    /// Slabs for the rows up to `upto` exist (allocated and zeroed).
    fn ensure_slabs(&mut self, upto: usize) -> Result<(), GpuError> {
        let (fw, fh) = self.spec.rendered;
        while self.allocated < upto.min(fh) {
            let y0 = self.allocated;
            let rows = self.slab_rows.min(fh - y0);
            let image = self.run((fw, rows), slab_job)?;
            self.rendered.pieces.push_back(Piece {
                y0,
                rows,
                image: Arc::new(image),
            });
            self.allocated = y0 + rows;
        }
        Ok(())
    }

    /// Samples and colours one tile: `w x h` pixels of the rendered picture at `off`, from `region`,
    /// camera RGB whose pixel (0, 0) is buffer pixel `origin` (the rectangle of the photo's buffer
    /// the tile reads, [`SamplerGrid`] says which). Tiles come in row order, left to right; a tile
    /// that reads nothing of the photo is simply not added (its pixels are black).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a tile outside the picture or in a row that has been
    /// advanced past, and the errors of the job.
    pub fn add_tile(
        &mut self,
        region: Arc<RgbImage>,
        origin: (usize, usize),
        off: (usize, usize),
        w: usize,
        h: usize,
    ) -> Result<(), GpuError> {
        let (fw, fh) = self.spec.rendered;
        if w == 0 || h == 0 || off.0 + w > fw || off.1 + h > fh {
            return Err(GpuError::InvalidInput(format!(
                "a {w} x {h} tile at {off:?} is outside the {fw} x {fh} picture"
            )));
        }
        if off.1 < self.rows_done {
            return Err(GpuError::InvalidInput(format!(
                "a tile at row {} arrives after the rows up to {} were declared complete",
                off.1, self.rows_done
            )));
        }
        self.ensure_slabs(off.1 + h)?;
        let rows = off.1..off.1 + h;
        let targets: Vec<SlabTarget> = self
            .rendered
            .overlapping(&rows)
            .into_iter()
            .map(|p| {
                let (a, b) = (rows.start.max(p.y0), rows.end.min(p.end()));
                SlabTarget {
                    image: Arc::clone(&p.image),
                    tile_row0: a - rows.start,
                    rows: b - a,
                    dst_x: off.0,
                    dst_y: a - p.y0,
                }
            })
            .collect();
        let data = TileData {
            region,
            origin,
            grid: self.spec.grid,
            off,
            w,
            h,
            chain: Arc::clone(&self.spec.chain),
            targets,
        };
        self.run(data, tile_job)
    }

    /// Declares the rows `0..rows` of the rendered picture complete and runs every stage that can:
    /// returns the bands that are finished, in order.
    ///
    /// # Errors
    ///
    /// The errors of the jobs; [`GpuError::Cancelled`] once the route's token is set.
    pub fn advance(&mut self, rows: usize) -> Result<Vec<Band>, GpuError> {
        let fh = self.spec.rendered.1;
        let (ow, oh) = self.spec.out;
        self.rows_done = self.rows_done.max(rows.min(fh));
        // Slabs for rows nobody drew on (the picture's last rows are all outside the photo) exist
        // and are black.
        self.ensure_slabs(self.rows_done)?;
        let mut bands = Vec::new();
        let resizes = self.taps.is_some();
        if let Some((htaps, vtaps)) = self.taps.clone() {
            // Horizontal pass over every slab that is complete.
            while let Some(slab) = self.rendered.pieces.front().cloned() {
                if slab.end() > self.rows_done {
                    break;
                }
                let mid = self.run(
                    HData {
                        taps: Arc::clone(&htaps),
                        slab: slab.clone(),
                    },
                    h_job,
                )?;
                self.mid.pieces.push_back(Piece {
                    y0: slab.y0,
                    rows: slab.rows,
                    image: Arc::new(mid),
                });
                self.mid_done = slab.end();
                self.rendered.pieces.pop_front();
            }
            // Vertical pass over every chunk of output rows whose input rows are all there.
            while self.next_v < oh {
                let n = self.v_rows.min(oh - self.next_v);
                let out_rows = self.next_v..self.next_v + n;
                let span = vtaps.taps().input_span(out_rows.clone());
                if span.end > self.mid_done {
                    break;
                }
                let pieces = self.mid.overlapping(&span);
                let out = self.run(
                    VData {
                        taps: Arc::clone(&vtaps),
                        pieces,
                        out_rows: out_rows.clone(),
                    },
                    v_job,
                )?;
                self.resized.pieces.push_back(Piece {
                    y0: out_rows.start,
                    rows: n,
                    image: Arc::new(out),
                });
                self.next_v = out_rows.end;
                // The next chunk starts at `next_v`; nothing before its first input row is needed.
                let keep_from = if self.next_v < oh {
                    vtaps.taps().input_span(self.next_v..self.next_v + 1).start
                } else {
                    fh
                };
                self.mid.release_below(keep_from);
            }
        }
        // C13 (and sharpening) over the finished rows of the written picture.
        let ready = if resizes { self.next_v } else { self.rows_done };
        loop {
            if self.next_band >= oh {
                break;
            }
            let (rows, window) = match &self.sharpen {
                Some((_, core, halo)) => {
                    let end = (self.next_band + core).min(oh);
                    let need = (end + halo).min(oh);
                    if need > ready {
                        break;
                    }
                    (
                        self.next_band..end,
                        self.next_band.saturating_sub(*halo)..need,
                    )
                }
                None => {
                    let end = (self.next_band + self.band_rows).min(oh);
                    // Wait for a whole band, except for the last one.
                    if end > ready {
                        break;
                    }
                    (self.next_band..end, self.next_band..end)
                }
            };
            let store = if resizes {
                &self.resized
            } else {
                &self.rendered
            };
            let data = BandData2 {
                pieces: store.overlapping(&window),
                rows: rows.clone(),
                window,
                sharpen: self.sharpen.as_ref().map(|(s, _, _)| s.clone()),
                format: self.spec.format,
                space: self.spec.space,
            };
            let bytes = self.run(data, band_job)?;
            bands.push(Band {
                y0: rows.start,
                rows: rows.len(),
                data: compact(self.spec.format, &bytes, ow, rows.len()),
            });
            self.next_band = rows.end;
            let halo = self.sharpen.as_ref().map_or(0, |(_, _, h)| *h);
            let keep_from = self.next_band.saturating_sub(halo);
            if resizes {
                self.resized.release_below(keep_from);
            } else {
                self.rendered.release_below(keep_from);
            }
        }
        Ok(bands)
    }

    /// Declares the whole picture complete and returns the last bands. Afterwards every row of the
    /// written picture has been encoded.
    ///
    /// # Errors
    ///
    /// As [`ExportTail::advance`]; [`GpuError::InvalidInput`] when rows are still missing at the end
    /// (a programming error of the caller).
    pub fn finish(&mut self) -> Result<Vec<Band>, GpuError> {
        let bands = self.advance(self.spec.rendered.1)?;
        if self.next_band != self.spec.out.1 {
            return Err(GpuError::InvalidInput(format!(
                "the export tail ended with {} of {} rows encoded",
                self.next_band, self.spec.out.1
            )));
        }
        Ok(bands)
    }
}

// ---------------------------------------------------------------------------------------------
// One tile, to the host (the DNG)
// ---------------------------------------------------------------------------------------------

struct HostTileData {
    region: Arc<RgbImage>,
    origin: (usize, usize),
    grid: SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
    chain: Arc<ColorChain>,
    stop: Stop,
}

fn host_tile_job(
    ctx: &GpuContext,
    token: &CancelToken,
    d: &HostTileData,
) -> Result<Vec<f32>, GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "tail host tile");
    let src = GpuImage::upload_rgb(ctx, "tail source", &d.region)?;
    let sampled = encode_sample(&mut passes, &src, d.origin, &d.grid, d.off, d.w, d.h)?;
    let prepared = PreparedChain::new(ctx, &d.chain)?;
    let out = GpuImage::alloc(
        ctx,
        "tail host tile",
        d.w,
        d.h,
        4,
        Placement::whole(d.w, d.h),
    )?;
    prepared.encode_stage(
        &mut passes,
        &sampled,
        Rect::new(0, 0, d.w, d.h),
        d.stop,
        &out,
        (0, 0),
    )?;
    let read = out.encode_read(&mut passes)?;
    token.check()?;
    let done = passes.submit()?.finish()?;
    let planar = out.take(&done, read)?;
    // Interleave on the device thread's side of the wait: three planes and the mask, a few
    // milliseconds for a tile.
    let n = d.w * d.h;
    let mut rgb = vec![0.0f32; 3 * n];
    for i in 0..n {
        if planar[3 * n + i] == 0.0 {
            rgb[3 * i] = planar[i];
            rgb[3 * i + 1] = planar[n + i];
            rgb[3 * i + 2] = planar[2 * n + i];
        }
    }
    Ok(rgb)
}

/// Samples one tile and runs the colour chain to `stop` ([`Stop::C12`] for a display-linear
/// preview, [`Stop::DngLinearAwg4`] for the pixels of a DNG), reads it back as interleaved f32 (zero
/// outside the photo), the layout of the CPU render's host frame. The DNG export uses it: its
/// frame stays on the host, because the white point (a quantile of the whole frame) is found there.
///
/// # Errors
///
/// The errors of the job; [`GpuError::Cancelled`] once the route's token is set.
#[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
pub fn colour_tile_to_host(
    route: &TailRoute<'_>,
    region: Arc<RgbImage>,
    origin: (usize, usize),
    grid: SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
    chain: Arc<ColorChain>,
    stop: Stop,
) -> Result<Vec<f32>, GpuError> {
    route.run(
        HostTileData {
            region,
            origin,
            grid,
            off,
            w,
            h,
            chain,
            stop,
        },
        host_tile_job,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_constants_are_consistent() {
        // A four-plane slab of the widest frame the plan names (9504 px, 61 MP) fits the design
        // binding at the nominal slab height; a three-plane gathered window of a sharpening chunk
        // does too.
        assert!(16 * 9504 * SLAB_ROWS as u64 <= SLAB_BYTES);
        assert!(16 * (SHARPEN_WINDOW_PIXELS as u64) <= SLAB_BYTES);
    }

    #[test]
    fn the_formats_choose_the_encodings() {
        assert_eq!(
            BandFormat::Eight { dither: true }.encoding(),
            Encoding::Rgb8 { background: [0; 3] }
        );
        assert_eq!(BandFormat::Sixteen.encoding(), Encoding::Rgb16 { bits: 16 });
        assert!(BandFormat::Eight { dither: true }.dither());
        assert!(!BandFormat::Eight { dither: false }.dither());
        assert!(!BandFormat::Sixteen.dither());
    }

    #[test]
    fn compaction_drops_the_row_padding() {
        // 5 pixels: 8-bit rows are padded to 8 pixels (24 bytes), 16-bit rows to 6 pixels (36 bytes).
        let w = 5;
        let mut padded8 = vec![0u8; 2 * 24];
        for r in 0..2 {
            for i in 0..w * 3 {
                padded8[r * 24 + i] = (r * 100 + i) as u8;
            }
            for b in &mut padded8[r * 24 + w * 3..(r + 1) * 24] {
                *b = 0xEE;
            }
        }
        let BandData::Eight(v) = compact(BandFormat::Eight { dither: true }, &padded8, w, 2) else {
            panic!("8-bit");
        };
        assert_eq!(v.len(), 2 * w * 3);
        assert!(!v.contains(&0xEE));
        assert_eq!(v[w * 3], 100);
        let mut padded16 = vec![0u8; 2 * 36];
        for r in 0..2 {
            for i in 0..w * 3 {
                let code = (r * 1000 + i) as u16;
                padded16[r * 36 + 2 * i..r * 36 + 2 * i + 2].copy_from_slice(&code.to_le_bytes());
            }
            for b in &mut padded16[r * 36 + w * 6..(r + 1) * 36] {
                *b = 0xEE;
            }
        }
        let BandData::Sixteen(v) = compact(BandFormat::Sixteen, &padded16, w, 2) else {
            panic!("16-bit");
        };
        assert_eq!(v.len(), 2 * w * 3);
        assert_eq!(v[0], 0);
        assert_eq!(v[w * 3], 1000);
        assert_eq!(v[2 * w * 3 - 1], 1000 + (w * 3 - 1) as u16);
    }
}

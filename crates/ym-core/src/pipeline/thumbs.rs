//! Thumbnails, look tiles, the histogram and the scheduled entry points.
//!
//! * **Thumbnails** (`thumbnail_ready`): the raw group at 1/8 (or the largest reduction that leaves
//!   at least twice the thumbnail size), geometry applied, the colour pass over at most 240 px,
//!   sRGB, PNG. They are rendered once after open (the session's [`ThumbnailRenderer`] hook,
//!   [`PhotoThumbnails`]) and again on look changes.
//! * **Look tiles** (`look_thumbnail`, an addition to `docs/01`): the same render with one look at
//!   100 %, for the tiles of the Look tab. Only the grade group differs between looks, so the raw
//!   group and the colour plan are shared by every tile.
//! * **Node thumbnails** (`node_thumbnails`, `docs/23` section 5.7): one picture of about
//!   [`NODE_THUMBNAIL_PX`] for every corrector of the photo, dead ones included, each the plan
//!   stopped after that node and then run through the Output, so that it looks like the photo the
//!   person knows. They are kept in memory in the photo's graph, by node and by a key that names
//!   everything up to the node, so a change late in the graph makes no new picture for the nodes
//!   before it.
//! * **Histogram** (`histogram`, an addition to `docs/01`): 64 bins of the LogC4 signal at the
//!   input of Main's curve (C9b, the warp's output: the curve is in front of the look since pipeline
//!   0.9.0), per channel, scaled so that the tallest bin is 1 (`docs/02` C11: "The histogram in the
//!   Curve tab is of LogC4 values").
//! * **Chroma scatter** (`chroma_scatter`, for the plane of the colour warp, `docs/11`): the photo's
//!   colours at the input of the warp (the C9 output), projected onto its chroma plane
//!   (`ym_color::warp::chroma_of`), from the same small view as the histogram, at most
//!   [`CHROMA_SCATTER_MAX_POINTS`] of them.
//! * **`at`** (`docs/01`, `docs/23` sections 7.5 and 9): the histogram, the scatter and Pick colour read the
//!   signal at a node. Left out it is Main's (as before); `input` is the Input's output; a corrector
//!   is read by the graph rewrite of the node thumbnails ([`settings_ending_at`], with no look and
//!   the node's own later stages taken out), so no kernel is involved (`signal_source`).
//! * **Scheduled calls**: `spawn_*` put a request on the render queue at the priority `docs/01`
//!   gives it and return a handle.
//! * **Apple Auto's picture** (`docs/19` section 3): the photo at its default settings through the
//!   Signature look at 100 % and the cine rendering, at 768 px on the long edge, 8-bit sRGB, on the CPU reference
//!   path ([`AdvisorPictures`], the session's [`PictureRenderer`] hook).

use std::borrow::Cow;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use ym_color::display::{ColorChain, OutputTarget, StageTrace};
use ym_color::logc::LogC4;
use ym_color::warp::chroma_of;
use ym_export::DetailStat;
use ym_gpu::queue::{CancelToken, JobError, JobHandle, Priority};
use ym_looks::library::LookId;

use super::RenderError;
use super::color::{
    LookProblem, LookProvider, PhotoSource, ResolvedLook, node_plan, raw_fingerprint, with_look,
};
use super::frame::{
    FrameOptions, FrameOutput, FullImage, FullRequest, RenderedFrame, Sampled, ViewSpec,
    render_view, sample_view_routed,
};
use super::geometry::Affine;
use super::graph::{Job, RenderOptions, Renderer};
use super::viewport::ViewportState;
use crate::apple_auto::{AdvisorPicture, PICTURE_LONG_EDGE_PX};
use crate::session::{PhotoId, PictureInput, PictureRenderer, ThumbnailInput, ThumbnailRenderer};
use crate::settings::{ColorWarp, Curves, DevelopSettings, NodeId, NodeLink, NodeRef};
use ym_color::look::LookTransform;
use ym_color::plan::Stop;

/// Bins of the histogram.
pub const HISTOGRAM_BINS: usize = 64;
/// Longest edge of the sampled image behind the histogram, in pixels.
const HISTOGRAM_LONG_EDGE_PX: usize = 192;

/// The `histogram` response of `src/api/types.ts`: per channel, [`HISTOGRAM_BINS`] values from black
/// to white, scaled so that the tallest bin of any channel is 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistogramData {
    /// The red channel.
    pub red: Vec<f64>,
    /// The green channel.
    pub green: Vec<f64>,
    /// The blue channel.
    pub blue: Vec<f64>,
}

/// The most points a [`ChromaScatter`] holds.
pub const CHROMA_SCATTER_MAX_POINTS: usize = 2048;
/// The scatter's coordinates are rounded to this step (the plane is drawn 352 px wide for 0.6 units,
/// so this is far below a pixel and keeps the answer small).
pub const CHROMA_SCATTER_STEP: f64 = 1e-4;

/// The `chroma_scatter` response of `src/api/types.ts`: the photo's colours at the input of the colour
/// warp (C9b), as points `(a, b)` of its chroma plane, flat: `[a0, b0, a1, b1, ...]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChromaScatter {
    /// At most [`CHROMA_SCATTER_MAX_POINTS`] points, each coordinate rounded to
    /// [`CHROMA_SCATTER_STEP`].
    pub points: Vec<f64>,
}

/// The pixels of a sampled view that are inside the photo, every `stride`-th of them so that at most
/// `max` are left: a fixed stride over the pixels in reading order, so the same view gives the same
/// points.
fn every_nth_inside(sampled: &Sampled, max: usize) -> Vec<usize> {
    let n = sampled.img.width * sampled.img.height;
    let outside = sampled.outside.as_deref();
    let inside: Vec<usize> = (0..n)
        .filter(|&i| !outside.is_some_and(|o| o[i] != 0))
        .collect();
    let stride = inside.len().div_ceil(max.max(1)).max(1);
    inside.into_iter().step_by(stride).collect()
}

/// A coordinate of the scatter: rounded to [`CHROMA_SCATTER_STEP`], never `-0`. Dividing the count
/// of steps by their number per unit gives the double nearest to the decimal, so the JSON says
/// `0.1234` and not `0.12340000000000001`.
fn scatter_coordinate(v: f64) -> f64 {
    const STEPS_PER_UNIT: f64 = 1e4;
    (v * STEPS_PER_UNIT).round() / STEPS_PER_UNIT + 0.0
}

/// Encodes 8-bit RGB as a PNG.
///
/// # Errors
///
/// [`RenderError::Image`] when the encoder refuses the size.
pub fn encode_png_rgb8(width: usize, height: usize, rgb: &[u8]) -> Result<Vec<u8>, RenderError> {
    let fail = |e: png::EncodingError| {
        RenderError::Image(format!("The thumbnail could not be written: {e}."))
    };
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width as u32, height as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(fail)?;
        w.write_image_data(rgb).map_err(fail)?;
    }
    Ok(out)
}

/// The size of a thumbnail of an image: the long edge at most `max_px`, never upscaled.
#[must_use]
pub fn thumbnail_size(image: (usize, usize), max_px: usize) -> (usize, usize) {
    let long = image.0.max(image.1).max(1);
    let s = (max_px as f64 / long as f64).min(1.0);
    (
        ((image.0 as f64 * s).round() as usize).max(1),
        ((image.1 as f64 * s).round() as usize).max(1),
    )
}

/// The reduction a thumbnail of `max_px` is made from: the largest of 8, 4 and 2 that leaves the
/// long edge at least twice the thumbnail, else full resolution.
fn thumbnail_scale(out: (usize, usize), max_px: usize) -> usize {
    let long = out.0.max(out.1);
    for k in [8usize, 4, 2] {
        if long / k >= 2 * max_px {
            return k;
        }
    }
    1
}

/// The long edge of a node's thumbnail, in pixels (`docs/23` section 5.7).
pub const NODE_THUMBNAIL_PX: usize = 160;

/// Counts up with every node thumbnail made in this run of the program, so that a picture has a
/// version nobody else has (an unchanged picture keeps its version when it is served again).
static NODE_THUMBNAIL_VERSION: AtomicU64 = AtomicU64::new(1);

/// One node's thumbnail ([`Renderer::node_thumbnails`]).
#[derive(Debug, Clone)]
pub struct NodeThumbnail {
    /// The corrector.
    pub node: u32,
    /// Names this picture: it changes when the picture is made again and stays while it is served
    /// from memory, so a caller that remembers what it showed knows whether the picture is new.
    pub version: u64,
    /// The PNG: sRGB, 8 bits, [`NODE_THUMBNAIL_PX`] on the long edge at most.
    pub png: Arc<Vec<u8>>,
    /// Whether this call made the picture (false: it was kept from an earlier call).
    pub fresh: bool,
}

/// What a photo's graph keeps of a node's thumbnail.
#[derive(Debug, Clone)]
pub(crate) struct NodeThumbnailEntry {
    /// What the picture was made from (see [`Renderer::node_thumbnails`]).
    pub(crate) key: String,
    pub(crate) version: u64,
    pub(crate) png: Arc<Vec<u8>>,
}

/// The settings of the graph that ends at `node`: the Output reads the node's picture, in place of
/// the one it read (the nodes after `node` become dead branches, and a dead `node` is live).
///
/// This is the plan stopped after the node's picture, then the Output: the plan of the result is
/// the plan compiled with [`Stop::Picture`] for `node` (a test compares the two), and the Output's
/// own settings (the look, the cine rendering, the sharpening) stay.
#[must_use]
pub fn settings_ending_at(settings: &DevelopSettings, node: u32) -> DevelopSettings {
    let mut s = settings.clone();
    let into_output = NodeRef::picture(NodeId::Output);
    let link = NodeLink::picture(NodeId::Corrector(node), NodeId::Output);
    if !s.nodes.links.contains(&link) {
        s.nodes.links.retain(|l| l.to != into_output);
        s.nodes.links.push(link);
    }
    s
}

/// Where in a node a panel reads its signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tap {
    /// The input of the node's curve: after its trim, its grade and its colour warp (the histogram,
    /// `docs/02` C11: "the histogram in the Curve tab is of LogC4 values").
    CurveInput,
    /// The input of the node's colour warp: after its trim and its grade (the scatter behind the
    /// pins and Pick colour, whose pins take hold of these colours).
    WarpInput,
}

/// The stage of a [`StageTrace`] that holds a signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Read {
    /// The Input's output (C8).
    InputOutput,
    /// Main's C9, after its trim and its grade.
    MainWarpInput,
    /// Main's C9b, after its colour warp.
    MainCurveInput,
    /// The last value of the chain (C10): the graph was rewritten so that this is the node's own
    /// stage.
    Last,
}

impl Read {
    /// The LogC4 value this stage holds in a trace.
    pub(crate) fn of(self, trace: &StageTrace) -> LogC4 {
        match self {
            Self::InputOutput => trace.c8_logc4,
            Self::MainWarpInput => trace.c9_logc4,
            Self::MainCurveInput => trace.c9b_logc4,
            Self::Last => trace.c10_logc4,
        }
    }
}

/// The photo to read, and the stage of its trace, for the signal at `at` (`None` is Main, node 1:
/// what the panels read before nodes).
///
/// * Main reads its own C9 or C9b, as always.
/// * The Input reads its output (C8).
/// * Another corrector `k` is read through a **graph rewrite**: the photo with the Output reading
///   node `k` ([`settings_ending_at`]; the nodes after it are dead branches), the look taken out
///   (Neutral, so that C10 is the last node's own value), and the node's later stages made identity
///   (its curve; for the warp's input also its warp), so that the chain's last value is the signal
///   at that stage of the node. Nothing in the kernels or the chain changes, and the raw group and
///   its caches are the photo's own.
///
/// # Errors
///
/// [`RenderError::Color`] for a node the photo does not have and for the Output, which has no
/// signal to read.
pub(crate) fn signal_source(
    src: &PhotoSource,
    at: Option<NodeId>,
    tap: Tap,
) -> Result<(Cow<'_, PhotoSource>, Read), RenderError> {
    let main = match tap {
        Tap::CurveInput => Read::MainCurveInput,
        Tap::WarpInput => Read::MainWarpInput,
    };
    match at {
        None => Ok((Cow::Borrowed(src), main)),
        Some(NodeId::Input) => Ok((Cow::Borrowed(src), Read::InputOutput)),
        Some(NodeId::Output) => Err(RenderError::Color(
            "The output has no picture to read here.".to_owned(),
        )),
        Some(NodeId::Corrector(1)) => Ok((Cow::Borrowed(src), main)),
        Some(NodeId::Corrector(k)) => {
            if src.settings.nodes.corrector(k).is_none() {
                return Err(RenderError::Color(format!("This photo has no node {k}.")));
            }
            let mut settings = settings_ending_at(&src.settings, k);
            settings.output_mut().look = LookId::neutral();
            if let Some(node) = settings.nodes.corrector_mut(k) {
                node.grade.curve = Curves::default();
                if tap == Tap::WarpInput {
                    node.grade.color_warp = ColorWarp::default();
                }
            }
            Ok((Cow::Owned(src.with_settings(settings)), Read::Last))
        }
    }
}

impl Renderer {
    fn thumbnail_spec(&self, job: &mut Job<'_>) -> Result<(ViewSpec, (usize, usize)), RenderError> {
        self.thumbnail_spec_at(job, self.options().thumbnail_px)
    }

    /// The view of a thumbnail whose long edge is at most `max_px`.
    fn thumbnail_spec_at(
        &self,
        job: &mut Job<'_>,
        max_px: usize,
    ) -> Result<(ViewSpec, (usize, usize)), RenderError> {
        let image = job.final_size()?;
        let (tw, th) = thumbnail_size(image, max_px);
        // The reduction follows the size of what is shown (a crop to a tenth of the photo needs a
        // finer image than the whole photo does).
        let k = thumbnail_scale(image, max_px);
        let spec = ViewSpec {
            out_w: tw,
            out_h: th,
            view_to_final: Affine::scale_translate(
                image.0 as f64 / tw as f64,
                image.1 as f64 / th as f64,
                0.0,
                0.0,
            ),
            k,
            target: OutputTarget::Srgb,
            before: false,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Thumbnail,
            dither_origin: (0, 0),
            output: FrameOutput::Host,
        };
        Ok((spec, (tw, th)))
    }

    /// The thumbnail's pixels: sRGB, 8 bits, `width * height * 3` interleaved, with the crop,
    /// straighten and rotation of the settings applied.
    fn thumbnail_rgb(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<(usize, usize, Vec<u8>), RenderError> {
        let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::Thumbnail);
        let (spec, (tw, th)) = self.thumbnail_spec(&mut job)?;
        let frame = render_view(job, &spec)?;
        let mut rgb = vec![0u8; 3 * tw * th];
        rgb.par_chunks_mut(3)
            .zip(frame.rgba.par_chunks(4))
            .for_each(|(d, s)| {
                d.copy_from_slice(&s[..3]);
            });
        Ok((tw, th, rgb))
    }

    /// The photo as a thumbnail: a PNG of at most 240 px on its long edge (`thumbnail_ready`), in
    /// sRGB, with the crop, straighten and rotation of the settings applied.
    ///
    /// The picture's detail statistic is kept for the size estimate of the Export panel
    /// ([`Renderer::thumbnail_detail`]); it always describes the newest thumbnail rendered here,
    /// so it follows the look, crop and turn the filmstrip follows.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn thumbnail_png(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RenderError> {
        let (tw, th, rgb) = self.thumbnail_rgb(src, cancel)?;
        self.note_thumbnail_detail(src.id, DetailStat::measure_rgb8(tw as u32, th as u32, &rgb));
        encode_png_rgb8(tw, th, &rgb)
    }

    /// A thumbnail for every corrector of the photo (`docs/23` section 5.7), in the order of the
    /// cards (the plan's order over all nodes, dead ones included): each is the photo with the plan
    /// stopped after that node's picture and then run through the Output
    /// ([`settings_ending_at`]), at most [`NODE_THUMBNAIL_PX`] on its long edge, in sRGB, as a PNG.
    ///
    /// The pictures are kept in the photo's graph under a key that holds the raw group, the
    /// geometry, the view, the Input's and the Output's settings and the plan up to the node
    /// (`GradeKey`: the nodes the plan runs to it, not the ones after it). A call finds every
    /// picture whose key is unchanged in memory and renders the others, on the device when the
    /// renderer has one and on the CPU otherwise ([`NodeThumbnail::fresh`] says which). Nothing
    /// goes to disk. The thumbnail does not touch the photo's detail statistic.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job. The pictures made before the error are
    /// kept.
    pub fn node_thumbnails(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Vec<NodeThumbnail>, RenderError> {
        let ids: Vec<u32> = node_plan(&src.settings.nodes, Stop::Output)?
            .cards
            .iter()
            .map(|card| card.id)
            .collect();
        // The view, and the part of every key that does not depend on the node: the raw group,
        // the geometry, the lens, the size and the output sharpening.
        let (spec, shared) = {
            let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::Thumbnail);
            let (spec, _) = self.thumbnail_spec_at(&mut job, NODE_THUMBNAIL_PX)?;
            let params = job.params()?;
            let mut shared = format!(
                "{:?}|{:?}|{:?}|{}x{}|{:?}|os{:?}",
                params.key(),
                job.geometry()?,
                spec.view_to_final,
                spec.out_w,
                spec.out_h,
                spec.k,
                src.settings.output().output_sharpening,
            );
            if let Some(lens) = job.lens_key()? {
                let _ = write!(shared, "|{lens}");
            }
            (spec, shared)
        };
        let graph = self.graph(src.id);
        graph.retain_node_thumbnails(&ids);
        let mut out = Vec::with_capacity(ids.len());
        for node in ids {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            let stopped = src.with_settings(settings_ending_at(&src.settings, node));
            let mut job = Job::new(self, &stopped, cancel)?.with_gpu_priority(Priority::Thumbnail);
            let key = format!("{shared}|{:?}", job.grade_key(spec.target, true)?);
            if let Some(kept) = graph.cached_node_thumbnail(node, &key) {
                out.push(NodeThumbnail {
                    node,
                    version: kept.version,
                    png: kept.png,
                    fresh: false,
                });
                continue;
            }
            let frame = render_view(job, &spec)?;
            let mut rgb = vec![0u8; 3 * spec.out_w * spec.out_h];
            rgb.par_chunks_mut(3)
                .zip(frame.rgba.par_chunks(4))
                .for_each(|(d, s)| {
                    d.copy_from_slice(&s[..3]);
                });
            let png = Arc::new(encode_png_rgb8(spec.out_w, spec.out_h, &rgb)?);
            let version = NODE_THUMBNAIL_VERSION.fetch_add(1, Ordering::Relaxed);
            graph.store_node_thumbnail(
                node,
                NodeThumbnailEntry {
                    key,
                    version,
                    png: Arc::clone(&png),
                },
            );
            out.push(NodeThumbnail {
                node,
                version,
                png,
                fresh: true,
            });
        }
        Ok(out)
    }

    /// Apple Auto's picture of a photo (`docs/19` section 3): `src` as it is (its settings, look and
    /// estimates), whole, at [`PICTURE_LONG_EDGE_PX`] on the long edge (a smaller photo is not
    /// enlarged), from the reduced raw group as a thumbnail is, in sRGB, 8 bits, `width * height * 3`
    /// bytes, rows top to bottom. It runs on the calling thread on whatever device this renderer has;
    /// [`AdvisorPictures`] keeps one without a device, so the picture is the CPU reference path's.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn advisor_picture(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<AdvisorPicture, RenderError> {
        let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::Thumbnail);
        let image = job.final_size()?;
        let (width, height) = thumbnail_size(image, PICTURE_LONG_EDGE_PX);
        let spec = ViewSpec {
            out_w: width,
            out_h: height,
            view_to_final: Affine::scale_translate(
                image.0 as f64 / width as f64,
                image.1 as f64 / height as f64,
                0.0,
                0.0,
            ),
            k: thumbnail_scale(image, PICTURE_LONG_EDGE_PX),
            target: OutputTarget::Srgb,
            before: false,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Thumbnail,
            dither_origin: (0, 0),
            // The model gets bytes in memory; this renderer has no device anyway.
            output: FrameOutput::Host,
        };
        let frame = render_view(job, &spec)?;
        let mut rgb = vec![0u8; 3 * width * height];
        rgb.par_chunks_mut(3)
            .zip(frame.rgba.par_chunks(4))
            .for_each(|(d, s)| {
                d.copy_from_slice(&s[..3]);
            });
        Ok(AdvisorPicture { width, height, rgb })
    }

    /// The photo as it is developed now, through one look at 100 % (the Look tab's tiles). It does
    /// not touch the photo's detail statistic: a tile is the photo through another look.
    ///
    /// # Errors
    ///
    /// [`RenderError::Look`] with "File missing or changed. Import again." when the look cannot be
    /// loaded, and the errors of [`Renderer::thumbnail_png`].
    pub fn look_thumbnail_png(
        &self,
        src: &PhotoSource,
        look: &LookId,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RenderError> {
        self.looks()
            .look(look)
            .map_err(|e| RenderError::Look(e.to_string()))?;
        let tile = src.with_settings(with_look(&src.settings, look));
        let (tw, th, rgb) = self.thumbnail_rgb(&tile, cancel)?;
        encode_png_rgb8(tw, th, &rgb)
    }

    /// The small view of the whole photo that the histogram and the chroma scatter read, and the
    /// grade chain of the photo as it is rendered now (of the graph that [`signal_source`] makes
    /// for `at`): `read` gets the sampled camera RGB (with the letterbox mask), the chain and the
    /// stage that holds the signal.
    fn panel_view<T>(
        &self,
        src: &PhotoSource,
        at: Option<NodeId>,
        tap: Tap,
        cancel: &CancelToken,
        read: impl FnOnce(&Sampled, &ColorChain, Read) -> T,
    ) -> Result<T, RenderError> {
        let (src, stage) = signal_source(src, at, tap)?;
        let mut job = Job::new(self, &src, cancel)?.with_gpu_priority(Priority::Thumbnail);
        let image = job.final_size()?;
        let (tw, th) = thumbnail_size(image, HISTOGRAM_LONG_EDGE_PX);
        let spec = ViewSpec {
            out_w: tw,
            out_h: th,
            view_to_final: Affine::scale_translate(
                image.0 as f64 / tw as f64,
                image.1 as f64 / th as f64,
                0.0,
                0.0,
            ),
            k: thumbnail_scale(image, HISTOGRAM_LONG_EDGE_PX),
            target: OutputTarget::Viewport,
            before: false,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Thumbnail,
            dither_origin: (0, 0),
            output: FrameOutput::Host,
        };
        let grade = job.grade(OutputTarget::Viewport, false)?;
        // The sample comes from the device group when the sampler is there; the reading stays on
        // the CPU (a few thousand pixels through the f64 trace: `docs/17-gpu.md` section 11).
        let view = sample_view_routed(&mut job, &spec)?;
        let out = read(&view.sampled, &grade.chain, stage);
        job.finish(
            view.scale,
            super::graph::ComputeDevice::Cpu,
            OutputTarget::Viewport,
        );
        Ok(out)
    }

    /// The histogram of the photo as rendered now (see the module documentation).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn histogram(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<HistogramData, RenderError> {
        self.histogram_at(src, None, cancel)
    }

    /// The histogram of the signal at a node (`docs/01`, the `at` argument): the input of its
    /// curve (after its trim, grade and colour warp), or the Input's output for `Input`; `None` is
    /// Main, which [`Renderer::histogram`] reads. See `signal_source`.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data, a node the photo does not have, the Output, or a
    /// cancelled job.
    pub fn histogram_at(
        &self,
        src: &PhotoSource,
        at: Option<NodeId>,
        cancel: &CancelToken,
    ) -> Result<HistogramData, RenderError> {
        let counts =
            self.panel_view(src, at, Tap::CurveInput, cancel, |sampled, chain, stage| {
                let img = &sampled.img;
                let n = img.width * img.height;
                let outside = sampled.outside.as_deref();
                (0..n)
                    .into_par_iter()
                    .fold(
                        || [[0u64; HISTOGRAM_BINS]; 3],
                        |mut acc, i| {
                            if outside.is_some_and(|o| o[i] != 0) {
                                return acc;
                            }
                            let p = img.pixel(i % img.width, i / img.width);
                            let t =
                                chain.trace([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                            for (c, acc_c) in acc.iter_mut().enumerate() {
                                let v = stage.of(&t).0[c].clamp(0.0, 1.0);
                                let bin =
                                    ((v * HISTOGRAM_BINS as f64) as usize).min(HISTOGRAM_BINS - 1);
                                acc_c[bin] += 1;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || [[0u64; HISTOGRAM_BINS]; 3],
                        |mut a, b| {
                            for c in 0..3 {
                                for i in 0..HISTOGRAM_BINS {
                                    a[c][i] += b[c][i];
                                }
                            }
                            a
                        },
                    )
            })?;
        let peak = counts.iter().flatten().copied().max().unwrap_or(0).max(1) as f64;
        let norm = |c: usize| -> Vec<f64> {
            counts[c]
                .iter()
                .map(|&v| ((v as f64 / peak) * 1000.0).round() / 1000.0)
                .collect()
        };
        Ok(HistogramData {
            red: norm(0),
            green: norm(1),
            blue: norm(2),
        })
    }

    /// The photo's colours at the input of the colour warp (the C9 output, `StageTrace::c9_logc4`),
    /// projected onto the warp's chroma plane with `ym_color::warp::chroma_of`: the faint scatter
    /// behind the pins (`docs/11` Colour warp). It reads the histogram's view (the whole photo at
    /// most 192 px on the long edge, the letterbox left out) and keeps every `n`-th pixel in reading
    /// order so that at most [`CHROMA_SCATTER_MAX_POINTS`] are left, so the same photo and settings
    /// give the same points; each coordinate is rounded to [`CHROMA_SCATTER_STEP`]. The warp itself
    /// does not move them (they are what the pins take hold of); everything before it does.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn chroma_scatter(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<ChromaScatter, RenderError> {
        self.chroma_scatter_at(src, None, cancel)
    }

    /// The scatter of the colours at the input of a node's colour warp (`docs/01`, the `at`
    /// argument): after its trim and its grade, or the Input's output for `Input`; `None` is Main,
    /// which [`Renderer::chroma_scatter`] reads. See `signal_source`.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data, a node the photo does not have, the Output, or a
    /// cancelled job.
    pub fn chroma_scatter_at(
        &self,
        src: &PhotoSource,
        at: Option<NodeId>,
        cancel: &CancelToken,
    ) -> Result<ChromaScatter, RenderError> {
        let points =
            self.panel_view(src, at, Tap::WarpInput, cancel, |sampled, chain, stage| {
                let img = &sampled.img;
                every_nth_inside(sampled, CHROMA_SCATTER_MAX_POINTS)
                    .par_iter()
                    .flat_map_iter(|&i| {
                        let p = img.pixel(i % img.width, i / img.width);
                        let t = chain.trace([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                        chroma_of(stage.of(&t)).map(scatter_coordinate)
                    })
                    .collect::<Vec<f64>>()
            })?;
        Ok(ChromaScatter { points })
    }

    // --- Scheduled calls -------------------------------------------------------------------

    /// Records that the user moved to `photo`: the viewport and tile jobs of the photo they left
    /// are cancelled; thumbnails and exports are not.
    pub fn navigate(&self, photo: PhotoId) -> usize {
        let left = self.note_current(photo);
        if let Some(old) = left {
            // Full-resolution intermediates (tiles, the prepared chain with its sigma map) live
            // only while the photo is being looked at; the reduced images stay for the filmstrip.
            if let Some(g) = self.existing_graph(old) {
                g.release_full_resolution();
            }
            // The device jobs of the viewport and tile renders of the photo left are cancelled
            // too, queued ones at once (they hold the photo's mosaic).
            self.cancel_gpu_navigation(old);
        }
        self.scheduler().navigate(photo)
    }

    /// Queues a viewport frame at [`Priority::Viewport`].
    #[must_use]
    pub fn spawn_viewport_frame(
        &self,
        src: PhotoSource,
        state: ViewportState,
        options: FrameOptions,
    ) -> JobHandle<Result<RenderedFrame, RenderError>> {
        self.spawn_viewport_frame_to(src, state, options, FrameOutput::Host)
    }

    /// [`Renderer::spawn_viewport_frame`] with the place the frame's bytes go
    /// ([`Renderer::viewport_frame_to`]): the native viewport asks for [`FrameOutput::Device`].
    #[must_use]
    pub fn spawn_viewport_frame_to(
        &self,
        src: PhotoSource,
        state: ViewportState,
        options: FrameOptions,
        output: FrameOutput,
    ) -> JobHandle<Result<RenderedFrame, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Viewport, Some(id), move |token| {
                r.viewport_frame_to(&src, &state, &options, output, token)
            })
    }

    /// Queues the full-resolution tiles a viewport shows at [`Priority::TileNeural`].
    #[must_use]
    pub fn spawn_tiles(
        &self,
        src: PhotoSource,
        state: ViewportState,
    ) -> JobHandle<Result<usize, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::TileNeural, Some(id), move |token| {
                r.prepare_tiles(&src, &state, token)
            })
    }

    /// Queues a thumbnail at [`Priority::Thumbnail`].
    #[must_use]
    pub fn spawn_thumbnail(&self, src: PhotoSource) -> JobHandle<Result<Vec<u8>, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Thumbnail, Some(id), move |token| {
                r.thumbnail_png(&src, token)
            })
    }

    /// Queues the node thumbnails of a photo ([`Renderer::node_thumbnails`]) at
    /// [`Priority::Thumbnail`]. Like a look tile it is a request of a panel for the photo on screen,
    /// and moving to another photo cancels it (see [`Renderer::spawn_look_thumbnail`]).
    #[must_use]
    pub fn spawn_node_thumbnails(
        &self,
        src: PhotoSource,
    ) -> JobHandle<Result<Vec<NodeThumbnail>, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            Priority::Thumbnail,
            id,
            move |token| r.node_thumbnails(&src, token),
            || Err(RenderError::Cancelled),
        )
    }

    /// Queues a look tile at [`Priority::Thumbnail`]. A tile is a request of the Look tab for the
    /// photo on screen: moving to another photo ([`Renderer::navigate`]) cancels it, pending or
    /// running, and a pending one lets go of the photo's decoded image at once; the handle then
    /// resolves with [`JobError::Cancelled`] (a running tile ends with
    /// [`RenderError::Cancelled`]; see
    /// [`JobScheduler::submit_superseded`](crate::scheduler::JobScheduler::submit_superseded)).
    #[must_use]
    pub fn spawn_look_thumbnail(
        &self,
        src: PhotoSource,
        look: LookId,
    ) -> JobHandle<Result<Vec<u8>, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            Priority::Thumbnail,
            id,
            move |token| r.look_thumbnail_png(&src, &look, token),
            || Err(RenderError::Cancelled),
        )
    }

    /// Queues the histogram of a photo at [`Priority::Thumbnail`]. Like a look tile it is a request
    /// of a panel for the photo on screen, and moving to another photo cancels it (see
    /// [`Renderer::spawn_look_thumbnail`]).
    #[must_use]
    pub fn spawn_histogram(
        &self,
        src: PhotoSource,
    ) -> JobHandle<Result<HistogramData, RenderError>> {
        self.spawn_histogram_at(src, None)
    }

    /// [`Renderer::spawn_histogram`] for the signal at a node ([`Renderer::histogram_at`]).
    #[must_use]
    pub fn spawn_histogram_at(
        &self,
        src: PhotoSource,
        at: Option<NodeId>,
    ) -> JobHandle<Result<HistogramData, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            Priority::Thumbnail,
            id,
            move |token| r.histogram_at(&src, at, token),
            || Err(RenderError::Cancelled),
        )
    }

    /// Queues the chroma scatter of a photo at [`Priority::Thumbnail`]. Like the histogram it is a
    /// request of a panel for the photo on screen, and moving to another photo cancels it (see
    /// [`Renderer::spawn_look_thumbnail`]).
    #[must_use]
    pub fn spawn_chroma_scatter(
        &self,
        src: PhotoSource,
    ) -> JobHandle<Result<ChromaScatter, RenderError>> {
        self.spawn_chroma_scatter_at(src, None)
    }

    /// [`Renderer::spawn_chroma_scatter`] for the colours at a node ([`Renderer::chroma_scatter_at`]).
    #[must_use]
    pub fn spawn_chroma_scatter_at(
        &self,
        src: PhotoSource,
        at: Option<NodeId>,
    ) -> JobHandle<Result<ChromaScatter, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            Priority::Thumbnail,
            id,
            move |token| r.chroma_scatter_at(&src, at, token),
            || Err(RenderError::Cancelled),
        )
    }

    /// Queues a full-resolution render at [`Priority::Export`]; navigation never cancels it. It runs
    /// on the export worker, which starts it when no viewport frame, tile pass or thumbnail is
    /// waiting and looks again before every tile, so browsing is not held up by an export
    /// ([`JobScheduler`](crate::scheduler::JobScheduler)).
    #[must_use]
    pub fn spawn_full(
        &self,
        src: PhotoSource,
        request: FullRequest,
    ) -> JobHandle<Result<FullImage, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Export, Some(id), move |token| {
                r.render_full_with(&src, &request, token, true, None)
            })
    }

    /// [`Renderer::spawn_full`] that also calls `progress` with the fraction of tiles done (0 to 1)
    /// after each tile, from the export worker's thread. The export queue shows it as the
    /// per-file percentage.
    #[must_use]
    pub fn spawn_full_reporting(
        &self,
        src: PhotoSource,
        request: FullRequest,
        progress: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> JobHandle<Result<FullImage, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Export, Some(id), move |token| {
                r.render_full_with(&src, &request, token, true, Some(&*progress))
            })
    }

    /// The hook the session calls to render a freshly developed photo's thumbnail: it queues the
    /// render at thumbnail priority and waits for it, so a viewport frame asked for meanwhile runs
    /// first.
    #[must_use]
    pub fn thumbnail_renderer(&self) -> Arc<dyn ThumbnailRenderer> {
        Arc::new(PhotoThumbnails {
            renderer: self.clone(),
        })
    }
}

/// The looks Apple Auto's picture is rendered with: the Signature look
/// (`<data>/looks/builtin/signature.json`, `SessionEnv::reference_look`), and Neutral. The picture's
/// display rendering is ours, the cine rendering.
struct PictureLooks {
    signature: Option<Arc<LookTransform>>,
}

impl LookProvider for PictureLooks {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        match (id.as_str(), &self.signature) {
            ("builtin:signature", Some(signature)) => Ok(ResolvedLook {
                transform: Arc::clone(signature),
                fingerprint: "builtin:signature".to_owned(),
            }),
            ("builtin:neutral", _) => Ok(ResolvedLook {
                transform: Arc::new(LookTransform::neutral()),
                fingerprint: "builtin:neutral".to_owned(),
            }),
            _ => Err(LookProblem(format!("The look {id} is not in the library."))),
        }
    }
}

/// The session's [`PictureRenderer`] hook: Apple Auto's pictures (`docs/19` section 3), made by a
/// renderer of their own that has no graphics device and only the Signature and Neutral looks, so the
/// app and `ym` give the model the same bytes on every Mac whatever the GPU policy and the look
/// library say. Each picture is the photo at its default
/// settings (every Auto the rule's: the photo's own estimates), no crop, no straightening, no turn by
/// the user and the camera's orientation applied, through the Signature look at 100 % (Neutral when
/// the data folder has none) and the cine rendering. Its caches are dropped after each picture.
pub struct AdvisorPictures {
    renderer: Renderer,
    look: LookId,
}

impl std::fmt::Debug for AdvisorPictures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdvisorPictures")
            .field("look", &self.look)
            .finish_non_exhaustive()
    }
}

impl AdvisorPictures {
    /// A picture renderer with the Signature look `signature` (`None`: Neutral) and the options of
    /// the app's renderer without its device (`options.gpu` is ignored).
    ///
    /// # Errors
    ///
    /// [`RenderError::Thread`] when the operating system cannot start the renderer's thread.
    pub fn new(
        signature: Option<Arc<LookTransform>>,
        options: &RenderOptions,
    ) -> Result<Self, RenderError> {
        let look = if signature.is_some() {
            LookId::builtin("signature")
        } else {
            LookId::neutral()
        };
        let options = RenderOptions {
            gpu: None,
            ..options.clone()
        };
        let renderer = Renderer::new(Arc::new(PictureLooks { signature }), options)?;
        Ok(Self { renderer, look })
    }

    /// The picture of a photo (see the type's documentation).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn picture(&self, input: &PictureInput<'_>) -> Result<AdvisorPicture, RenderError> {
        if (input.cancelled)() {
            return Err(RenderError::Cancelled);
        }
        let src = PhotoSource {
            id: input.id,
            raw: Arc::clone(input.raw),
            profile: Arc::clone(input.profile),
            estimates: Arc::clone(input.estimates),
            settings: DevelopSettings::with_look(self.look.clone()),
            pick: None,
            fingerprint: raw_fingerprint(input.raw),
            sensor: input.camera.matched.sensor,
            capture_method: super::raw::DEFAULT_CAPTURE_METHOD,
            before_view: false,
        };
        let picture = self.renderer.advisor_picture(&src, &CancelToken::new());
        // One picture per photo: nothing of it is worth keeping.
        self.renderer.forget(input.id);
        picture
    }
}

impl PictureRenderer for AdvisorPictures {
    fn render(&self, input: &PictureInput<'_>) -> Result<AdvisorPicture, String> {
        self.picture(input).map_err(|e| e.to_string())
    }
}

/// The session's thumbnail hook backed by a [`Renderer`] (see [`Renderer::thumbnail_renderer`]).
#[derive(Debug, Clone)]
pub struct PhotoThumbnails {
    renderer: Renderer,
}

impl ThumbnailRenderer for PhotoThumbnails {
    fn render(&self, input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String> {
        let src = PhotoSource::from_thumbnail_input(input);
        let handle = self.renderer.spawn_thumbnail(src);
        loop {
            if (input.cancelled)() {
                handle.cancel();
                return Err("The session was closed.".to_owned());
            }
            match handle.wait_timeout(Duration::from_millis(25)) {
                None => continue,
                Some(Ok(Ok(png))) => return Ok(png),
                Some(Ok(Err(e))) => return Err(e.to_string()),
                Some(Err(JobError::Cancelled)) => {
                    return Err("The render was cancelled.".to_owned());
                }
                Some(Err(e)) => return Err(e.to_string()),
            }
        }
    }
}

#[cfg(test)]
mod scatter_tests {
    use ym_gpu::cpu::RgbImage;

    use super::*;

    #[test]
    fn the_scatter_keeps_every_nth_pixel_inside_the_photo_and_at_most_the_limit() {
        // 10 x 10 with the first row on the paper: 90 pixels inside.
        let mut outside = vec![0u8; 100];
        outside[..10].fill(1);
        let sampled = Sampled {
            img: RgbImage::new(10, 10),
            outside: Some(outside),
        };
        let all = every_nth_inside(&sampled, 1000);
        assert_eq!(all, (10..100).collect::<Vec<_>>());
        // 90 pixels for at most 40: every third, 30 of them, from the first inside.
        let some = every_nth_inside(&sampled, 40);
        assert_eq!(some.len(), 30);
        assert_eq!(&some[..3], &[10, 13, 16]);
        assert_eq!(every_nth_inside(&sampled, 45).len(), 45);
        // No mask: every pixel is inside.
        let whole = Sampled {
            img: RgbImage::new(10, 10),
            outside: None,
        };
        assert_eq!(every_nth_inside(&whole, 2048).len(), 100);
    }

    #[test]
    fn scatter_coordinates_are_the_nearest_decimals_of_four_places() {
        assert_eq!(scatter_coordinate(0.123_449), 0.1234);
        assert_eq!(scatter_coordinate(-0.123_46), -0.1235);
        assert_eq!(
            serde_json::to_string(&[scatter_coordinate(0.123_4), scatter_coordinate(0.3)]).unwrap(),
            "[0.1234,0.3]"
        );
        // Never `-0`.
        assert!(scatter_coordinate(-0.000_01).is_sign_positive());
    }
}

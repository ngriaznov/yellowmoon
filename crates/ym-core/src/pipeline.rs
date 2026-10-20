//! The render pipeline: decode, R1 to R9, C1 to C13, geometry and the preview strategy
//! (`docs/01-architecture.md` sections Render graph, Preview strategy, Scheduling and threads;
//! `docs/02-colour-pipeline.md`; `docs/05-raw-processing.md`).
//!
//! # Shape
//!
//! A [`Renderer`] is the shared engine: it knows where looks come from ([`color::LookProvider`]),
//! which display rendering is chosen, whether a GPU is available for the per-pixel colour stages,
//! and it owns one [`graph::PhotoGraph`] (the stage caches) per photo. Every call takes a
//! [`color::PhotoSource`], a snapshot of one photo (decoded image, profile, estimates, settings),
//! so a settings change is just a new source: the graph compares cache keys and recomputes only the
//! groups the change reaches.
//!
//! | Group | Stages | Cached as | Key |
//! |---|---|---|---|
//! | Decode | black, white, clip mask | [`raw::FrameData`] (the area copy) | the photo |
//! | Raw | R1, R3 to R9 | an `RgbImage` per scale, or the tiles at 100 % | `decode.*`, noise reduction, sharpening, the gains `g0`, the noise model |
//! | Colour | C1 to C8 | [`color::ColorPlan`] (fused matrix, tone) | light, white balance, pick, exposure, `rolloff` (the highlight range and the toe) |
//! | Grade | C9 to C13 (C9b, the colour warp, included) | [`color::GradePlan`] (the prepared `ColorChain`) | everything else (the colour warp's pins as the chain prepares them, the white lift of C12 included: the automatic amount of the photo's estimates and `white_offset_ev`, [`color::white_lift`]), and the colour key |
//! | Geometry | crop, straighten, orientation | nothing: applied when sampling | `crop`, `orientation` |
//! | Output | output sharpening of the finished picture, then C13 | the display-linear frame of a view ([`frame`] module documentation) | `output_sharpening` (and every group above) |
//!
//! The raw group is the expensive one (seconds at full resolution); the others are matrices and
//! tables. Colour and grade work per pixel, so a colour edit re-runs one pass over the viewport's
//! pixels ([`Renderer::viewport_frame`]) on the GPU when there is one and on the CPU otherwise;
//! both paths produce the same pixels within the parity tolerances of `ym-gpu`.
//!
//! # Previews (`docs/01` section Preview strategy)
//!
//! * **Fit view**: the raw group at 1/2 or 1/4 resolution (binned, classical), chosen so that the
//!   image has at least as many pixels as the viewport shows (never upscaled), then the colour pass
//!   over the viewport's device pixels.
//! * **100 % and above**: full-resolution tiles of 512 px through `cpu::chain` (the neural path
//!   replaces them when trained weights exist; see [`NeuralStatus`]); until a tile is ready the
//!   frame can be served from the fit image ([`FrameOptions::coarse_only`]).
//! * **Thumbnails and look tiles**: the raw group at 1/8 (or 1/2 for small photos), the colour pass
//!   over at most 240 px, PNG bytes.
//! * **Export**: [`Renderer::render_full`], full resolution, tile by tile.
//!
//! # Determinism
//!
//! The same photo, settings and version give the same pixels whatever the number of threads: every
//! parallel loop splits by output row and no value is reduced across rows (`tests/determinism.rs`).

use serde::{Deserialize, Serialize};
use ym_export::ColorSpace;

pub mod color;
pub mod frame;
pub mod geometry;
pub mod graph;
pub mod raw;
pub mod thumbs;
pub mod viewport;

pub use color::{DisplayChoice, LookProblem, LookProvider, NeutralOnly, PhotoSource, ResolvedLook};
pub use frame::{
    ChromaSample, DeviceFrame, FitFrame, FrameOptions, FrameOutput, FullImage, FullRequest,
    FullTarget, PixelSample, PixelWindow, RenderedFrame, ViewportFrame,
};
pub use graph::{
    ComputeDevice, PhotoGraph, RawPrefetch, RenderOptions, RenderReport, Renderer, StageCounters,
    StageRuns, StageTimings,
};
pub use thumbs::{AdvisorPictures, ChromaScatter, HistogramData, PhotoThumbnails};
pub use viewport::{ViewportInput, ViewportMode, ViewportRect, ViewportState};

/// Why a render failed. The text is for the person using the app: what went wrong and what to do.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RenderError {
    /// The job was cancelled (navigation, a newer request, the session closed).
    #[error("The render was cancelled.")]
    Cancelled,
    /// The photo's data cannot be rendered.
    #[error("{0}")]
    Image(String),
    /// The colour settings cannot make a chain (a refused Pick white, a bad curve).
    #[error("{0}")]
    Color(String),
    /// A look or display LUT cannot be used.
    #[error("{0}")]
    Look(String),
    /// The render would not fit in the memory budget.
    #[error(
        "There is not enough memory to render this photo at full size. Close other apps or export a smaller size."
    )]
    Memory,
    /// The GPU failed in a way the CPU path could not make up for.
    #[error("The graphics card could not render this photo: {0}")]
    Gpu(String),
    /// The job scheduler has shut down.
    #[error("Yellowmoon is shutting down.")]
    Shutdown,
    /// The render thread could not be started.
    #[error(
        "Yellowmoon could not start its render thread ({0}). Close other apps and open it again."
    )]
    Thread(String),
}

impl RenderError {
    /// Wraps an error of the `ym_gpu::cpu` stages.
    pub(crate) fn from_cpu(e: ym_gpu::cpu::CpuError) -> Self {
        Self::Image(format!("This photo can't be rendered: {e}."))
    }
}

/// Which demosaic produced the picture (the header's "Neural · 42 ms · Display P3").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderPath {
    /// The learned joint demosaic and denoise.
    Neural,
    /// The classical demosaic (the adaptive ratio demosaic by default, `docs/05` R6) and bilateral denoise.
    Classical,
}

impl RenderPath {
    /// The word the status text uses.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Neural => "Neural",
            Self::Classical => "Classical",
        }
    }
}

/// The `render_stats` event payload of `src/api/types.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderStats {
    /// Milliseconds the last render took.
    pub ms: f64,
    /// The demosaic path.
    pub path: RenderPath,
    /// The output colour space.
    pub space: ColorSpace,
    /// The zoom of the frame in percent: device pixels per pixel of the final image times 100, not
    /// rounded (`docs/01` Native viewport, "Zoom"). Only a viewport frame has one.
    #[serde(default)]
    pub zoom_percent: Option<f64>,
    /// The zoom at which the whole final image fits the cell, in percent and not rounded: what the
    /// header's "Fit 24%" and the decision whether a fixed zoom shows the whole picture are made
    /// from. Only a viewport frame has one.
    #[serde(default)]
    pub fit_percent: Option<f64>,
}

/// How far a viewport frame is zoomed: device pixels per pixel of the final image (after the lens
/// crop, orientation, straighten and crop, at full resolution), as `ViewportState::zoom` and
/// `ViewportState::fit_zoom` work them out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameZoom {
    /// The zoom in effect.
    pub zoom: f64,
    /// The zoom at which the whole final image fits the cell.
    pub fit_zoom: f64,
}

impl RenderStats {
    /// The header's status text, for example `Classical · 42 ms · Display P3`.
    #[must_use]
    pub fn status_text(&self) -> String {
        format!(
            "{} · {} ms · {}",
            self.path.label(),
            self.ms.round() as i64,
            self.space.display_name()
        )
    }
}

/// Whether the learned demosaic can run, and why not.
///
/// No trained weights exist yet (`docs/07`, M2-06), so today the answer is always no and every
/// render says so: the classical path runs instead and [`RenderReport::notes`] records the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuralStatus {
    /// Whether the network can run.
    pub available: bool,
    /// The reason it cannot, in plain words.
    pub reason: Option<String>,
}

impl NeuralStatus {
    /// The status when no weights are installed.
    #[must_use]
    pub fn unavailable() -> Self {
        Self {
            available: false,
            reason: Some(
                "No trained demosaic model is installed yet, so the classical demosaic was used."
                    .to_owned(),
            ),
        }
    }

    /// Looks for trained weights under `<data_root>/models/jdd/`. The runtime is wired in by
    /// `docs/15` M2-07; until it is, weights alone do not make the path available.
    #[must_use]
    pub fn probe(data_root: Option<&std::path::Path>) -> Self {
        let weights = data_root.map(|d| d.join("models").join("jdd"));
        match weights {
            Some(dir)
                if dir.is_dir()
                    && std::fs::read_dir(&dir).is_ok_and(|mut it| {
                        it.any(|e| {
                            e.ok().is_some_and(|e| {
                                e.path().extension().is_some_and(|x| x == "safetensors")
                            })
                        })
                    }) =>
            {
                Self {
                    available: false,
                    reason: Some(
                        "The learned demosaic is not connected to the renderer in this build, so the classical demosaic was used."
                            .to_owned(),
                    ),
                }
            }
            _ => Self::unavailable(),
        }
    }
}

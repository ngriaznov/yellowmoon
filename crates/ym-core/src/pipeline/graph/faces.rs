//! The Face Refinement node in the render graph (`docs/27-skin-retouch.md` section 10, S-11b, decisions 9 and 13).
//!
//! The node retouches the faces of the photo. Finding them, and the models that make their masks and blemish maps, are
//! expensive (a face's mesh and segmenter take 250 to 690 ms) and must not run again when a slider moves, so they run on a
//! picture that depends on no node's grade, no amount and no look: the **analysis picture**
//! ([`face_source`]): the Input's colour in the photo's final frame (crop, turn and lens kept, so the faces are where the
//! render has them), the Neutral look, 8 bits, sRGB, at most [`FACE_ANALYSIS_LONG_EDGE_PX`] on the long edge, made by the
//! renderer of its own that has no graphics device, so the picture, the faces and the masks are the CPU reference path's
//! bytes whatever the machine and the app and `ym` agree. The step that runs on every frame (`frame/film_pass.rs`) only
//! applies the cached masks and blemish maps to the node's input (`ym_retouch::retouch_skin_region`).
//!
//! The pattern is the depth map's (`graph/depth.rs`):
//!
//! * **The analysis** ([`FaceEntry`]: the numbered faces, their boxes and the prepared `RetouchCache`) lives in the photo's
//!   graph ([`FaceSlot`]), in memory, and never touches the disk. Its key names what the picture was made from
//!   ([`Renderer::faces_key`]): the raw group's numbers, the Input's colour and geometry (never a node's grade, the look or an
//!   amount), the weights and the rules' version.
//! * **The worker** ([`Renderer::faces_prepare`]) makes it in the background as soon as the core sees a Face Refinement node
//!   on the photo the person is looking at: one worker per photo that waits [`PREPARE_DEBOUNCE`], runs the picture and the
//!   networks, and tells the listener ([`FaceState`]).
//! * **A viewer never waits.** A frame takes the analysis the photo has for its picture; for another picture's (the Input's
//!   exposure moved) it keeps the latest one while the geometry is the same, and a photo with none shows the node's input
//!   unchanged until the listener tells the page to draw again. A failure is kept for its key and nothing asks again until
//!   the key changes.
//! * **An export makes its own analysis** when the photo has none of the right key, on the CPU, and does not keep it, so its
//!   bytes depend on nothing but the file and the settings.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ym_color::plan::Stop as PlanStop;
use ym_gpu::queue::CancelToken;
use ym_retouch::{Face, LinearAwg4Image, RetouchCache, SkinSettings};

use super::object::fnv;
use super::{Job, PhotoGraph, Renderer, lock};
use crate::pipeline::RenderError;
use crate::pipeline::color::{PhotoSource, color_key, live_face_nodes};
use crate::pipeline::faces::{
    FACE_ANALYSIS_LONG_EDGE_PX, FACES_FORMAT_VERSION, FACES_RULES_VERSION, FaceBox, FaceEvent,
    FaceModel, FaceState, select_faces,
};
use crate::session::PhotoId;
use crate::settings::DevelopSettings;

/// The side of a face's picture in the Faces card, in pixels.
pub const FACE_CROP_PX: usize = 104;
/// The picture's side over the larger side of the face's box.
pub const FACE_CROP_SCALE: f64 = 1.4;

/// The square of `FACE_CROP_SCALE` times the larger side of `rect` (`x, y, width, height`, pixels of the picture), centred on
/// it and moved to lie inside the picture (smaller than that only when the picture is), as `(x0, y0, side)`.
fn crop_square(size: (usize, usize), rect: (i32, i32, u32, u32)) -> (f64, f64, f64) {
    let (w, h) = (size.0 as f64, size.1 as f64);
    let side = (FACE_CROP_SCALE * f64::from(rect.2.max(rect.3)))
        .min(w.min(h))
        .max(1.0);
    let (cx, cy) = (
        f64::from(rect.0) + f64::from(rect.2) / 2.0,
        f64::from(rect.1) + f64::from(rect.3) / 2.0,
    );
    let x0 = (cx - side / 2.0).clamp(0.0, w - side);
    let y0 = (cy - side / 2.0).clamp(0.0, h - side);
    (x0, y0, side)
}

/// Cuts the face's square out of an sRGB 8-bit picture and makes it `FACE_CROP_PX` square: each pixel is the mean of a grid
/// of bilinear samples as fine as the reduction needs, so a big face is not read one pixel in ten.
fn face_crop_rgb(picture: &[u8], size: (usize, usize), rect: (i32, i32, u32, u32)) -> Vec<u8> {
    let (x0, y0, side) = crop_square(size, rect);
    let step = side / FACE_CROP_PX as f64;
    let taps = (step.ceil() as usize).clamp(1, 8);
    let at = |x: f64, y: f64, c: usize| -> f64 {
        // Pixel centres at integers + 0.5.
        let (fx, fy) = (
            (x - 0.5).clamp(0.0, (size.0 - 1) as f64),
            (y - 0.5).clamp(0.0, (size.1 - 1) as f64),
        );
        let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - ix as f64, fy - iy as f64);
        let (jx, jy) = ((ix + 1).min(size.0 - 1), (iy + 1).min(size.1 - 1));
        let p = |x: usize, y: usize| f64::from(picture[(y * size.0 + x) * 3 + c]);
        (p(ix, iy) * (1.0 - tx) + p(jx, iy) * tx) * (1.0 - ty)
            + (p(ix, jy) * (1.0 - tx) + p(jx, jy) * tx) * ty
    };
    let mut out = vec![0u8; 3 * FACE_CROP_PX * FACE_CROP_PX];
    for oy in 0..FACE_CROP_PX {
        for ox in 0..FACE_CROP_PX {
            for c in 0..3 {
                let mut sum = 0.0;
                for ty in 0..taps {
                    for tx in 0..taps {
                        let x = x0 + (ox as f64 + (tx as f64 + 0.5) / taps as f64) * step;
                        let y = y0 + (oy as f64 + (ty as f64 + 0.5) / taps as f64) * step;
                        sum += at(x, y, c);
                    }
                }
                out[(oy * FACE_CROP_PX + ox) * 3 + c] =
                    (sum / (taps * taps) as f64).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// How long the background worker waits before it starts, so that a drag of the Input's exposure (which asks for a new
/// analysis on every tick) makes one analysis, not a hundred (the same wait as the depth map's).
pub(crate) const PREPARE_DEBOUNCE: Duration = super::object::PREPARE_DEBOUNCE;

/// How often a request that waits for the worker looks at its own cancel flag.
const WAIT_STEP: Duration = Duration::from_millis(40);

/// Who hears when the face worker starts or ends its work for a photo ([`Renderer::set_faces_listener`]).
pub(crate) type FacesListener = Arc<dyn Fn(PhotoId, FaceEvent) + Send + Sync>;

/// The photo as the faces are looked for in it: the Input's picture in the photo's **final frame**. The settings keep the
/// Input's raw, light, noise, crop, turn and lens fields and drop everything that is a later step or a view of the result: no
/// node but Main at its identity, the Neutral look and no output sharpening, so grading a node, changing an amount or the
/// look, or moving a slider of another node never moves a face. The crop, the turn and the lens stay because the faces are
/// placed in the frame the node's picture has.
#[must_use]
pub(crate) fn face_source(src: &PhotoSource) -> PhotoSource {
    use crate::settings::NodeGrade;
    use ym_looks::library::LookId;
    let mut s = DevelopSettings::default();
    *s.input_mut() = src.settings.input().clone();
    *s.main_grade_mut() = NodeGrade::neutral();
    let out = s.output_mut();
    out.look = LookId::neutral();
    out.look_intensity = 100.0;
    out.output_sharpening = 0.0;
    src.with_settings(s)
}

/// What an analysis is made from: the text names the picture, the weights and the rules; `frame` names the geometry alone (a
/// stale analysis is kept for a viewer only while the frame is the same).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacesKey {
    /// The analysis' key.
    pub text: String,
    /// The geometry's part of the key: the final frame's size, the crop, the turn and the lens.
    pub frame: String,
}

/// One face of an analysis.
#[derive(Debug, Clone)]
pub(crate) struct FoundFace {
    /// 1 for the leftmost.
    pub number: u32,
    /// The face in pixels of the analysis picture.
    pub face: Face,
}

/// Where the time of one analysis went, in milliseconds (the log and the measurements of `docs/27` decision 13).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FaceTimings {
    /// The analysis picture, rendered on the CPU reference path.
    pub picture_ms: f64,
    /// The finder.
    pub find_ms: f64,
    /// The masks and blemish maps of every face (`prepare_skin`).
    pub prepare_ms: f64,
    /// All of it.
    pub total_ms: f64,
    /// The faces kept.
    pub faces: usize,
}

/// How many regions' device plans an analysis keeps ([`FaceEntry::device_plans`]).
pub(crate) const PLANS_KEPT: usize = 16;

/// What a region's device plan is made for: the frame's size, the region and the faces with their settings (the analysis' own
/// masks and maps are the entry's).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlanKey {
    frame: (usize, usize),
    area: ym_color::mask_raster::Area,
    faces: Vec<(Face, SkinSettings)>,
}

impl PlanKey {
    pub(crate) fn new(
        frame: (usize, usize),
        area: ym_color::mask_raster::Area,
        faces: Vec<(Face, SkinSettings)>,
    ) -> Self {
        Self { frame, area, faces }
    }
}

/// The photo's faces with the masks and blemish maps made for them.
#[derive(Debug)]
pub(crate) struct FaceEntry {
    /// What the analysis was made from.
    pub(crate) key: FacesKey,
    /// The FNV-1a of the key: the grade cache key names the faces by it.
    pub(crate) hash: u64,
    /// The analysis picture's size: the frame the faces, masks and maps are in.
    pub(crate) analysis_px: (u32, u32),
    /// The numbered faces, left to right.
    pub(crate) faces: Vec<FoundFace>,
    /// The analysis picture as the finder saw it: sRGB, 8 bits, interleaved, `analysis_px` big. It shows the Input's colour
    /// without any node's grade (the faces' pictures in the panel, [`Renderer::face_crop_png`]).
    pub(crate) srgb8: Arc<Vec<u8>>,
    /// The same picture as the masks and maps were made on: linear AWG4, `analysis_px` big. The face step compares its values with
    /// the node's input to find the tone curve upstream (`ym_retouch::ToneRelation`, `docs/27` S-12).
    pub(crate) linear: Arc<LinearAwg4Image>,
    /// The masks and blemish maps, prepared; read only from here on.
    pub(crate) retouch: RetouchCache,
    /// Where the time of the analysis went.
    pub(crate) timings: FaceTimings,
    /// The device's plans of the regions made for this analysis (`frame/face_pass.rs`, `FaceWork::device_node`): the CPU's plan of
    /// a region and the analysis' quantiles are the same for every frame of the same size and settings, so a slider of another node,
    /// a pan or a tile of an export finds them here. The latest few, least recently used out.
    pub(crate) device_plans: Mutex<Vec<(PlanKey, ym_gpu::kernels::film_stack::FaceRegionSpec)>>,
}

impl FaceEntry {
    /// The faces as the panel lists them: boxes in the oriented photo frame, as shares of its size.
    pub(crate) fn boxes(&self) -> Vec<FaceBox> {
        let (w, h) = (
            f64::from(self.analysis_px.0.max(1)),
            f64::from(self.analysis_px.1.max(1)),
        );
        self.faces
            .iter()
            .map(|f| FaceBox {
                number: f.number,
                x: f64::from(f.face.rect.x_px) / w,
                y: f64::from(f.face.rect.y_px) / h,
                width: f64::from(f.face.rect.width_px) / w,
                height: f64::from(f.face.rect.height_px) / h,
            })
            .collect()
    }

    /// The faces of the analysis, each with the settings `settings_of` gives for its number.
    pub(crate) fn with_settings(
        &self,
        settings_of: impl Fn(u32) -> SkinSettings,
    ) -> Vec<(Face, SkinSettings)> {
        self.faces
            .iter()
            .map(|f| (f.face, settings_of(f.number)))
            .collect()
    }
}

/// One run of the analysis.
struct Running {
    key: String,
    stop: CancelToken,
    /// Started in the background: leaving the photo or a newer wish stops it. A run an export asked for is never stopped by
    /// anyone but its caller.
    speculative: bool,
}

/// A wish of the background worker.
struct Wanted {
    key: FacesKey,
    src: PhotoSource,
    /// When the wish was made: the worker starts once nobody has changed it for [`PREPARE_DEBOUNCE`].
    at: Instant,
}

#[derive(Default)]
struct SlotState {
    /// The latest analysis the photo has. A request for another picture replaces it when it finishes.
    ready: Option<Arc<FaceEntry>>,
    /// The analysis of this key failed, and why (the sentence). Nothing asks again until the key changes.
    failed: Option<(String, String)>,
    /// The analysis is running for this key.
    running: Option<Running>,
    /// What the background worker will make next.
    wanted: Option<Wanted>,
    /// A background worker is alive for the photo (it may be waiting out the debounce).
    worker: bool,
    /// The worker is making an analysis: the node says "Finding faces…".
    finding: bool,
    /// How many times the analysis finished for the photo.
    analyses: u64,
    /// How many workers were started for the photo.
    workers: u64,
}

/// What a photo's graph keeps for its Face Refinement nodes.
#[derive(Default)]
pub(crate) struct FaceSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

impl FaceSlot {
    /// Bytes the analysis holds: the masks (about a byte a pixel of each face's region, five masks) and the maps (16 bytes a
    /// cell), estimated from the faces' boxes.
    pub(crate) fn bytes(&self) -> usize {
        lock(&self.state).ready.as_ref().map_or(0, |e| {
            e.faces
                .iter()
                .map(|f| {
                    let r = f.face.rect;
                    let area = r.width_px as usize * r.height_px as usize;
                    area * 9 * 2
                })
                .sum()
        })
    }

    /// The person left the photo: nothing is wanted for it, and a run that was started in the background stops.
    pub(crate) fn abandon(&self) {
        let mut st = lock(&self.state);
        st.wanted = None;
        if let Some(run) = st.running.as_ref().filter(|r| r.speculative) {
            run.stop.cancel();
        }
    }

    fn latest(&self) -> Option<Arc<FaceEntry>> {
        lock(&self.state).ready.clone()
    }

    fn failure(&self, key: &str) -> Option<String> {
        lock(&self.state)
            .failed
            .as_ref()
            .filter(|(k, _)| k == key)
            .map(|(_, why)| why.clone())
    }
}

/// What a request remembers of its Face Refinement nodes.
#[derive(Default)]
pub(crate) struct JobFaces {
    key: Option<FacesKey>,
    entry: Option<Arc<FaceEntry>>,
    /// Looked for already (the answer, `None` included, is final for the request).
    asked: bool,
}

impl JobFaces {
    /// The analysis this request uses, when it has one.
    pub(crate) fn entry(&self) -> Option<&Arc<FaceEntry>> {
        self.entry.as_ref()
    }
}

/// The LogC4 / AWG4 picture (interleaved `f32`) as linear AWG4: what `ym-retouch` takes (the node's own input is decoded the same
/// way, `face_pass`).
fn linear_awg4(width: usize, height: usize, logc4: &[f32]) -> Result<LinearAwg4Image, RenderError> {
    use rayon::prelude::*;
    let data: Vec<[f32; 3]> = logc4
        .par_chunks_exact(3)
        .map(|p| {
            let l = ym_color::logc::logc4_decode_rgb([
                f64::from(p[0]),
                f64::from(p[1]),
                f64::from(p[2]),
            ]);
            [l[0] as f32, l[1] as f32, l[2] as f32]
        })
        .collect();
    LinearAwg4Image::new(width as u32, height as u32, data)
        .map_err(|e| RenderError::Mask(format!("Faces can't be found: {e}.")))
}

// ---------------------------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------------------------

impl Renderer {
    /// The face model this renderer has, when its weights are there and have not been found damaged.
    #[must_use]
    pub fn face_model(&self) -> Option<Arc<FaceModel>> {
        self.options().face_model.clone().filter(|m| m.available())
    }

    /// Whether this renderer can run a Face Refinement node.
    #[must_use]
    pub fn faces_available(&self) -> bool {
        self.face_model().is_some()
    }

    /// Who hears when the face worker starts or ends its work for a photo. Replaces the one set before.
    pub fn set_faces_listener(&self, f: FacesListener) {
        *lock(&self.inner.faces_listener) = Some(f);
    }

    fn faces_notify(&self, id: PhotoId, event: FaceEvent) {
        let listener = lock(&self.inner.faces_listener).clone();
        if let Some(f) = listener {
            f(id, event);
        }
    }

    /// Whether the worker is finding the faces of the photo ("Finding faces…").
    #[must_use]
    pub fn faces_working(&self, id: PhotoId) -> bool {
        self.existing_graph(id)
            .is_some_and(|g| lock(&g.faces.state).finding)
    }

    /// Whether a face worker thread is alive for the photo, even one that only waits out the debounce.
    #[must_use]
    pub fn faces_worker_alive(&self, id: PhotoId) -> bool {
        self.existing_graph(id)
            .is_some_and(|g| lock(&g.faces.state).worker)
    }

    /// Where the time of the photo's latest analysis went, when it has one.
    #[must_use]
    pub fn faces_timings(&self, id: PhotoId) -> Option<FaceTimings> {
        self.existing_graph(id)
            .and_then(|g| g.faces.latest())
            .map(|e| e.timings)
    }

    /// How many times a render left a face region unretouched although pixels it was asked for lie in it, since this renderer
    /// was made. It must stay 0: a half-retouched face is a visible defect (`docs/27` S-11b2).
    #[must_use]
    pub fn face_regions_cut(&self) -> u64 {
        self.inner
            .face_regions_cut
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// A render left a region unretouched that the pixels it was asked for meet (logged, and counted for the tests).
    pub(crate) fn note_face_region_cut(&self) {
        self.inner
            .face_regions_cut
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// How many times the analysis finished for the photo since its graph was made.
    #[must_use]
    pub fn faces_analyses(&self, id: PhotoId) -> u64 {
        self.existing_graph(id)
            .map_or(0, |g| lock(&g.faces.state).analyses)
    }

    /// How many background workers were started for the photo since its graph was made.
    #[must_use]
    pub fn faces_workers(&self, id: PhotoId) -> u64 {
        self.existing_graph(id)
            .map_or(0, |g| lock(&g.faces.state).workers)
    }

    /// Whether the photo has an analysis for the picture of these settings (a frame would not wait).
    #[must_use]
    pub fn faces_ready(&self, src: &PhotoSource) -> bool {
        let Ok(key) = self.faces_key(src) else {
            return false;
        };
        self.existing_graph(src.id).is_some_and(|g| {
            lock(&g.faces.state)
                .ready
                .as_ref()
                .is_some_and(|e| e.key.text == key.text)
        })
    }

    /// What the panel of a Face Refinement node says about the photo's faces now: where the analysis is and the faces it has
    /// found for the picture of these settings. Asking makes no graph and changes nothing.
    #[must_use]
    pub fn face_state(&self, src: &PhotoSource) -> FaceState {
        if self.face_model().is_none() {
            return FaceState::models_missing();
        }
        let Ok(key) = self.faces_key(src) else {
            return FaceState::finding();
        };
        let Some(graph) = self.existing_graph(src.id) else {
            return FaceState::finding();
        };
        let st = lock(&graph.faces.state);
        if let Some(e) = st.ready.as_ref().filter(|e| e.key.text == key.text) {
            return FaceState::ready(e.boxes());
        }
        if let Some((_, why)) = st.failed.as_ref().filter(|(k, _)| *k == key.text) {
            return FaceState::failed(why.clone());
        }
        FaceState::finding()
    }

    /// The picture of face `number` (1-based) for the Faces card: a square around the face's box, `FACE_CROP_SCALE` times its
    /// larger side, moved to stay inside the frame, cut from the analysis picture (the Input's colour without any retouch) and
    /// made `FACE_CROP_PX` square, as a PNG.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] with a plain sentence before the faces are known and for a number the photo does not have.
    pub fn face_crop_png(&self, src: &PhotoSource, number: u32) -> Result<Vec<u8>, RenderError> {
        let not_yet =
            || RenderError::Mask("The faces are not found yet. Try again in a moment.".to_owned());
        if self.face_model().is_none() {
            return Err(RenderError::Mask(
                crate::pipeline::faces::NO_FACE_MODEL_SENTENCE.to_owned(),
            ));
        }
        let key = self.faces_key(src)?;
        let entry = self
            .existing_graph(src.id)
            .and_then(|g| {
                lock(&g.faces.state)
                    .ready
                    .clone()
                    .filter(|e| e.key.text == key.text)
            })
            .ok_or_else(not_yet)?;
        let face = entry
            .faces
            .iter()
            .find(|f| f.number == number)
            .ok_or_else(|| {
                RenderError::Mask(format!(
                    "This photo has no face {number}. It has {}.",
                    entry.faces.len()
                ))
            })?;
        let rgb = face_crop_rgb(
            &entry.srgb8,
            (entry.analysis_px.0 as usize, entry.analysis_px.1 as usize),
            (
                face.face.rect.x_px,
                face.face.rect.y_px,
                face.face.rect.width_px,
                face.face.rect.height_px,
            ),
        );
        crate::pipeline::thumbs::encode_png_rgb8(FACE_CROP_PX, FACE_CROP_PX, &rgb)
    }

    /// Queues [`Renderer::face_crop_png`] at the thumbnails' priority: it is a request of a panel for the photo on screen, and
    /// moving to another photo cancels it (as for a look tile, [`Renderer::spawn_look_thumbnail`]).
    #[must_use]
    pub fn spawn_face_crop(
        &self,
        src: PhotoSource,
        number: u32,
    ) -> ym_gpu::queue::JobHandle<Result<Vec<u8>, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            ym_gpu::queue::Priority::Thumbnail,
            id,
            move |_token| r.face_crop_png(&src, number),
            || Err(RenderError::Cancelled),
        )
    }

    /// What the photo's analysis is made from: the Input's picture in the final frame (the photo, the raw group's numbers,
    /// the Input's colour, and the crop, turn and lens; nothing a node, an amount, the look or the output sharpening says),
    /// the weights and the rules. Equal keys are equal analyses.
    ///
    /// # Errors
    ///
    /// [`RenderError::Color`] for estimates that cannot make a picture, and [`RenderError::Mask`] when the renderer has no
    /// model.
    pub fn faces_key(&self, src: &PhotoSource) -> Result<FacesKey, RenderError> {
        let model = self.face_model().ok_or_else(|| {
            RenderError::Mask(crate::pipeline::faces::NO_FACE_MODEL_SENTENCE.to_owned())
        })?;
        let reference = self.object_reference()?;
        let stripped = face_source(src);
        let cancel = CancelToken::new();
        let mut job = Job::new(reference, &stripped, &cancel)?;
        let raw = job.params()?.key();
        let size = job.final_size()?;
        let frame = format!(
            "{size:?}|{:?}|{:?}",
            job.geometry()?,
            job.lens_key()?.unwrap_or_default()
        );
        let text = format!(
            "{:016x}|{raw:?}|{:?}|{frame}|faces|{}|f{FACES_FORMAT_VERSION}|r{FACES_RULES_VERSION}|{}px",
            stripped.fingerprint,
            color_key(&stripped),
            model.weights_id(),
            FACE_ANALYSIS_LONG_EDGE_PX,
        );
        Ok(FacesKey { text, frame })
    }

    /// Makes the photo's analysis for `key`: the picture on the CPU reference path, the finder, the numbering and the masks
    /// and blemish maps of every face. `cancel` stops the picture and is looked at between the stages.
    fn make_faces_entry(
        &self,
        model: &FaceModel,
        src: &PhotoSource,
        key: &FacesKey,
        cancel: &CancelToken,
        also: &(dyn Fn() -> bool + Sync),
    ) -> Result<FaceEntry, RenderError> {
        let stopped = || cancel.is_cancelled() || also();
        let started = Instant::now();
        let pictures = self.face_pictures_made(src, cancel)?;
        let picture_ms = started.elapsed().as_secs_f64() * 1000.0;
        if stopped() {
            return Err(RenderError::Cancelled);
        }
        let full = pictures.full;
        let analysis_px = (pictures.width as u32, pictures.height as u32);
        let full_px = (full.0 as u32, full.1 as u32);
        let found = model.find(
            &pictures.srgb8,
            pictures.width,
            pictures.height,
            full.0.max(full.1),
        )?;
        let find_ms = started.elapsed().as_secs_f64() * 1000.0 - picture_ms;
        if stopped() {
            return Err(RenderError::Cancelled);
        }
        let numbered = select_faces(&found, analysis_px, full_px, model.min_face_px());
        let hash = fnv(&key.text);
        let faces: Vec<FoundFace> = numbered
            .iter()
            .map(|&(number, face)| FoundFace { number, face })
            .collect();
        let mut retouch = RetouchCache::new(hash, 0);
        let prepare_started = Instant::now();
        let image = linear_awg4(pictures.width, pictures.height, &pictures.logc4)?;
        if !faces.is_empty() {
            let models = model.models()?;
            let with_settings: Vec<(Face, SkinSettings)> = faces
                .iter()
                .map(|f| (f.face, SkinSettings::default()))
                .collect();
            model
                .on_threads(|| {
                    ym_retouch::prepare_skin(&image, &with_settings, &models, &mut retouch)
                })
                .map_err(|e| RenderError::Mask(format!("Faces can't be found: {e}.")))?;
        }
        if stopped() {
            return Err(RenderError::Cancelled);
        }
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(
            ms,
            picture_ms,
            find_ms,
            prepare_ms,
            faces = faces.len(),
            "the faces (the picture, the finder and the masks)"
        );
        Ok(FaceEntry {
            key: key.clone(),
            hash,
            analysis_px,
            timings: FaceTimings {
                picture_ms,
                find_ms,
                prepare_ms,
                total_ms: ms,
                faces: faces.len(),
            },
            faces,
            srgb8: Arc::new(pictures.srgb8),
            linear: Arc::new(image),
            retouch,
            device_plans: Mutex::default(),
        })
    }

    /// The photo's analysis for `key`, made when it has none: waits for the run when it is running for this key, stops a run
    /// that was started in the background for another one, and runs the analysis itself when nobody does.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` is cancelled (or a run of this call was stopped because the person left the
    /// photo and the call is a background one), and the errors of the picture and the model.
    fn ensure_faces_entry(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &FaceModel,
        src: &PhotoSource,
        key: &FacesKey,
        speculative: bool,
        cancel: &CancelToken,
    ) -> Result<Arc<FaceEntry>, RenderError> {
        let slot = &graph.faces;
        loop {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            let mut st = lock(&slot.state);
            if let Some(ready) = st.ready.as_ref().filter(|e| e.key.text == key.text) {
                return Ok(Arc::clone(ready));
            }
            if let Some(run) = &st.running {
                if run.key != key.text && run.speculative && !speculative {
                    run.stop.cancel();
                }
                let (guard, _) = slot
                    .cv
                    .wait_timeout(st, WAIT_STEP)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                drop(guard);
                continue;
            }
            let stop = CancelToken::new();
            st.running = Some(Running {
                key: key.text.clone(),
                stop: stop.clone(),
                speculative,
            });
            drop(st);
            // `stop` is cancelled by the person leaving the photo; `cancel` by the caller.
            let result = self.make_faces_entry(model, src, key, &stop, &|| cancel.is_cancelled());
            let mut st = lock(&slot.state);
            st.running = None;
            match result {
                Ok(entry) => {
                    let entry = Arc::new(entry);
                    st.ready = Some(Arc::clone(&entry));
                    st.failed = None;
                    st.analyses += 1;
                    slot.cv.notify_all();
                    return Ok(entry);
                }
                Err(RenderError::Cancelled) => {
                    slot.cv.notify_all();
                    drop(st);
                    if speculative || cancel.is_cancelled() {
                        return Err(RenderError::Cancelled);
                    }
                }
                Err(e) => {
                    st.failed = Some((key.text.clone(), e.to_string()));
                    slot.cv.notify_all();
                    return Err(e);
                }
            }
        }
    }

    /// Asks for the photo's analysis in the background (`docs/27` decision 9: when a Face Refinement node appears on the
    /// photo). Returns at once. Nothing is done when the model is missing, the photo has the analysis, it is being made, it
    /// failed for this picture, or a worker already wants it; a worker waits until nobody has changed the wish for a moment
    /// (150 ms) so a stream of requests makes one analysis.
    pub fn faces_prepare(&self, src: &PhotoSource) {
        let Some(model) = self.face_model() else {
            return;
        };
        let Ok(key) = self.faces_key(src) else {
            return;
        };
        let graph = self.graph(src.id);
        self.faces_prepare_keyed(&graph, &model, src, key);
    }

    fn faces_prepare_keyed(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &Arc<FaceModel>,
        src: &PhotoSource,
        key: FacesKey,
    ) {
        {
            let mut st = lock(&graph.faces.state);
            if st.ready.as_ref().is_some_and(|r| r.key.text == key.text)
                || st.failed.as_ref().is_some_and(|(k, _)| *k == key.text)
                || st.running.as_ref().is_some_and(|r| r.key == key.text)
                || st.wanted.as_ref().is_some_and(|w| w.key == key)
            {
                return;
            }
            if let Some(run) = st.running.as_ref().filter(|r| r.speculative) {
                run.stop.cancel();
            }
            st.wanted = Some(Wanted {
                key,
                src: src.clone(),
                at: Instant::now(),
            });
            if st.worker {
                return;
            }
            st.worker = true;
            st.workers += 1;
        }
        let renderer = self.clone();
        let (worker_graph, worker_model) = (Arc::clone(graph), Arc::clone(model));
        let spawned = std::thread::Builder::new()
            .name("yellowmoon-faces-prepare".to_owned())
            .spawn(move || renderer.faces_worker(&worker_graph, &worker_model));
        if let Err(e) = spawned {
            tracing::error!(error = %e, "the face worker could not start");
            let mut st = lock(&graph.faces.state);
            st.worker = false;
            st.wanted = None;
        }
    }

    /// The background worker of a photo: makes what is wanted, one picture at a time, until nothing is. A wish is taken once
    /// it has been left alone for [`PREPARE_DEBOUNCE`], so a stream of them makes one analysis, of the last picture. The
    /// listener hears [`FaceState::finding`] when the first run begins and the outcome when the worker ends; a worker whose
    /// wishes were all dropped before a run began (the person left the photo) ends without a word.
    fn faces_worker(&self, graph: &Arc<PhotoGraph>, model: &Arc<FaceModel>) {
        let id = graph.id();
        let mut announced = false;
        let mut outcome: Option<Result<Arc<FaceEntry>, String>> = None;
        let mut last: Option<PhotoSource> = None;
        loop {
            let wanted = {
                let mut st = lock(&graph.faces.state);
                match &st.wanted {
                    None => {
                        st.worker = false;
                        st.finding = false;
                        break;
                    }
                    Some(w) if w.at.elapsed() < PREPARE_DEBOUNCE => {
                        let wait = PREPARE_DEBOUNCE - w.at.elapsed();
                        drop(st);
                        std::thread::sleep(wait);
                        continue;
                    }
                    Some(_) => st.wanted.take(),
                }
            };
            let Some(wanted) = wanted else { continue };
            if self.is_retired(id) {
                continue;
            }
            lock(&graph.faces.state).finding = true;
            if !announced {
                announced = true;
                self.faces_notify(id, FaceEvent::Started);
            }
            last = Some(wanted.src.clone());
            match self.ensure_faces_entry(
                graph,
                model,
                &wanted.src,
                &wanted.key,
                true,
                &CancelToken::new(),
            ) {
                Ok(e) => {
                    tracing::debug!(
                        ms = e.timings.total_ms,
                        faces = e.faces.len(),
                        "the faces are ready"
                    );
                    outcome = Some(Ok(e));
                }
                Err(RenderError::Cancelled) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "the faces could not be found");
                    outcome = Some(Err(e.to_string()));
                }
            }
        }
        if announced {
            let state = match (outcome, last) {
                (Some(Ok(e)), _) => FaceState::ready(e.boxes()),
                (Some(Err(why)), _) => FaceState::failed(why),
                // Every run was stopped (the person went to another photo, or the picture changed again): the state the
                // photo has now, so that a `Started` always has its `Finished`.
                (None, Some(src)) => self.face_state(&src),
                (None, None) => FaceState::finding(),
            };
            self.faces_notify(id, FaceEvent::Finished(state));
        }
    }

    /// The photo is no longer the one being looked at: the worker stops working for it if it started on its own.
    pub(crate) fn faces_leave(&self, id: PhotoId) {
        if let Some(g) = self.existing_graph(id) {
            g.faces.abandon();
        }
    }

    /// The photo's analysis for the picture of these settings, waiting for the worker when it runs and making it when nobody
    /// does: for checks and diagnostics.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when there is no model or the analysis fails; [`RenderError::Cancelled`]; the errors of the
    /// picture.
    pub fn faces_found(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Vec<FaceBox>, RenderError> {
        let model = self.face_model().ok_or_else(|| {
            RenderError::Mask(crate::pipeline::faces::NO_FACE_MODEL_SENTENCE.to_owned())
        })?;
        let key = self.faces_key(src)?;
        let graph = self.graph(src.id);
        if let Some(why) = graph.faces.failure(&key.text) {
            return Err(RenderError::Mask(why));
        }
        Ok(self
            .ensure_faces_entry(&graph, &model, src, &key, false, cancel)?
            .boxes())
    }
}

// ---------------------------------------------------------------------------------------------
// A request
// ---------------------------------------------------------------------------------------------

impl Job<'_> {
    /// The photo's analysis this request found for its Face Refinement nodes ([`Job::faces_for`] ran), if it has one.
    pub(crate) fn faces_entry(&self) -> Option<Arc<FaceEntry>> {
        self.faces.entry().cloned()
    }

    /// The analysis the Show skin areas view draws from (`docs/27` S-11c): the photo's latest one while the photo's frame is the
    /// same, whether or not a node is live. A view never waits and never starts the worker (the node's own panel did when the
    /// node was added); `None` before the faces are known or without models.
    pub(crate) fn faces_for_view(&mut self) -> Option<Arc<FaceEntry>> {
        self.opts.face_model.as_ref().filter(|m| m.available())?;
        let key = match &self.faces.key {
            Some(k) => k.clone(),
            None => {
                let k = self.r.faces_key(self.src).ok()?;
                self.faces.key = Some(k.clone());
                k
            }
        };
        self.g.faces.latest().filter(|e| e.key.frame == key.frame)
    }

    /// A render left a face region unretouched that its pixels meet ([`Renderer::face_regions_cut`]).
    pub(crate) fn note_face_region_cut(&self) {
        self.r.note_face_region_cut();
    }

    /// The photo's face view of this request ([`Job::prepare_mask_view`] found it): the face number and its analysis.
    pub(crate) fn face_view(&self) -> Option<&(u32, Arc<FaceEntry>)> {
        self.face_view.as_ref()
    }

    /// The photo's analysis for the plan that ends at `stop`, when that plan has a live Face Refinement node. `None` for a
    /// plan with none, when this machine has no model, and (for a viewer) when the photo has no analysis yet: the worker is
    /// started in the background and the node passes its picture through until the listener tells the page to draw again. A
    /// viewer keeps the latest analysis of another picture meanwhile when the photo's frame is the same. An export makes
    /// the analysis itself when the photo has none of the right key (and does not keep it).
    ///
    /// # Errors
    ///
    /// For an export, [`RenderError::Mask`] with "Faces can't be found: …"; [`RenderError::Cancelled`]; the errors of the
    /// picture.
    pub(crate) fn faces_for(
        &mut self,
        settings: &DevelopSettings,
        stop: PlanStop,
    ) -> Result<Option<Arc<FaceEntry>>, RenderError> {
        if live_face_nodes(settings, stop).is_empty() {
            return Ok(None);
        }
        let Some(model) = self.opts.face_model.clone().filter(|m| m.available()) else {
            return Ok(None);
        };
        if self.faces.asked {
            return Ok(self.faces.entry.clone());
        }
        let key = match &self.faces.key {
            Some(k) => k.clone(),
            None => {
                let k = self.r.faces_key(self.src)?;
                self.faces.key = Some(k.clone());
                k
            }
        };
        let ready = self.g.faces.latest();
        if let Some(e) = ready.as_ref().filter(|e| e.key.text == key.text) {
            self.faces.asked = true;
            self.faces.entry = Some(Arc::clone(e));
            return Ok(Some(Arc::clone(e)));
        }
        if self.restartable {
            self.check()?;
            let cancel = self.cancel;
            let made =
                Arc::new(
                    self.r
                        .make_faces_entry(&model, self.src, &key, cancel, &|| false)?,
                );
            self.faces.asked = true;
            self.faces.entry = Some(Arc::clone(&made));
            return Ok(Some(made));
        }
        // A viewer: the worker makes it (for the photo being looked at; the thumbnails of the folder do not start it).
        if self.g.faces.failure(&key.text).is_none() && self.r.is_protected(self.src.id) {
            self.r
                .faces_prepare_keyed(&self.g, &model, self.src, key.clone());
        }
        // Meanwhile the latest analysis, while the photo's frame is the same: the faces stay put while the picture moves.
        let stale = ready.filter(|e| e.key.frame == key.frame);
        self.faces.asked = true;
        self.faces.entry = stale.clone();
        Ok(stale)
    }
}

#[cfg(test)]
mod crop_tests {
    use super::*;

    #[test]
    fn the_square_is_one_point_four_times_the_larger_side_and_stays_in_the_picture() {
        // A box of 100 x 160 in a 1000 x 800 picture: side 224, centred.
        let (x0, y0, side) = crop_square((1000, 800), (400, 300, 100, 160));
        assert!((side - 224.0).abs() < 1e-9);
        assert!((x0 - (450.0 - 112.0)).abs() < 1e-9 && (y0 - (380.0 - 112.0)).abs() < 1e-9);
        // Against the top-left corner it moves, it does not shrink.
        let (x0, y0, side) = crop_square((1000, 800), (0, 0, 100, 160));
        assert_eq!((x0, y0), (0.0, 0.0));
        assert!((side - 224.0).abs() < 1e-9);
        // Against the bottom-right one.
        let (x0, y0, side) = crop_square((1000, 800), (900, 700, 100, 100));
        assert!((x0 + side - 1000.0).abs() < 1e-9 && (y0 + side - 800.0).abs() < 1e-9);
        // A face as big as the picture: the square is the picture's short side.
        let (_, _, side) = crop_square((1000, 800), (0, 0, 1000, 800));
        assert!((side - 800.0).abs() < 1e-9);
    }

    #[test]
    fn the_crop_is_104_px_square_and_holds_the_faces_pixels() {
        // A picture whose red channel is x / 4: a crop around a box reads the columns the square covers.
        let (w, h) = (500usize, 400usize);
        let mut picture = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                picture[(y * w + x) * 3] = (x / 2).min(255) as u8;
                picture[(y * w + x) * 3 + 1] = (y / 2).min(255) as u8;
            }
        }
        let rgb = face_crop_rgb(&picture, (w, h), (200, 100, 100, 100));
        assert_eq!(rgb.len(), FACE_CROP_PX * FACE_CROP_PX * 3);
        // The square: side 140, from x 180 and y 80. Its first pixel is near (180, 80), its last near (320, 220).
        let first = (f64::from(rgb[0]), f64::from(rgb[1]));
        assert!(
            (first.0 - 180.0 / 2.0).abs() < 1.5 && (first.1 - 80.0 / 2.0).abs() < 1.5,
            "{first:?}"
        );
        let last = &rgb[(FACE_CROP_PX * FACE_CROP_PX - 1) * 3..];
        assert!(
            (f64::from(last[0]) - 320.0 / 2.0).abs() < 1.5
                && (f64::from(last[1]) - 220.0 / 2.0).abs() < 1.5,
            "{last:?}"
        );
    }
}

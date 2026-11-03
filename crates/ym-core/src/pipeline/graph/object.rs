//! The Object mask in the render graph (`docs/23-nodes.md` section 4.4, `docs/algorithms/06-masks.md` section 6).
//!
//! An Object row's raster `S0` is `Decoded::soft_mask` of SAM's decoder run on the row's stored points and box, at 1024 px
//! on the long edge. The decoder reads the photo's **features**: the encoder's output for the Input's picture
//! ([`object_source`]: the photo frame, the Input's colour, the Neutral look, 8 bits, sRGB, no output sharpening),
//! rendered by a renderer of its own that has no graphics device, so the picture, the features and the mask are the CPU
//! reference path's bytes whatever the machine, and the app and `ym` agree.
//!
//! * **The features** (about 16 MB) live in the photo's graph ([`ObjectSlot`]), so they last as long as the renderer keeps the
//!   photo (three photos) and never touch the disk. Their key names what the picture was made from ([`Renderer::object_key`]):
//!   the raw group's numbers and the Input's colour, never a node's grade or the look.
//! * **The encoder runs in the background** ([`Renderer::object_prepare`]) as soon as the core sees an Object row on the photo the
//!   person is looking at: one worker per photo that waits a moment (a slider drag would otherwise start an encode on every
//!   tick), makes the features and tells the listener ([`ObjectPhase`]). A run that was started in the background stops when the
//!   person leaves the photo or asks for features of another picture (the encoder checks between its blocks).
//! * **`mask_find`** ([`Renderer::mask_find`]) adds the click or the box to the row, waits for the encoder if it is running,
//!   decodes, and answers the row's new points and box. Only a found mask is answered; the plane it made is kept
//!   ([`ObjectSlot`]), so the render that follows does not run the decoder again.
//! * **A render** takes the planes of the Object rows from [`Job::objects_for`]. A viewer never waits for the encoder: it draws
//!   the latest features it has (the mask stays where it was while the Input's exposure moves), or nothing, and the listener
//!   tells the page to draw again when the new features are ready. An export makes the features itself when the photo has none
//!   of the right key, so its bytes depend on nothing but the file and the settings.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ym_color::mask_raster::SourcePlane;
use ym_color::plan::Stop as PlanStop;
use ym_gpu::queue::CancelToken;
use ym_ml::model::MlError;
use ym_ml::sam2::{Decoding, Features, Picture};

use super::{Job, PhotoGraph, Renderer, lock};
use crate::pipeline::RenderError;
use crate::pipeline::color::{PhotoSource, color_key, object_rows, unmakeable_mask_with};
use crate::pipeline::object::{
    MAX_POINTS, MaskFound, NothingFound, ObjectModel, ObjectPhase, ObjectPlanes, ObjectPrompt,
    asks_for_mask, object_source, plane_of, prompt_key, round_box, round_coordinate, sam_prompt,
};
use crate::session::PhotoId;
use crate::settings::{
    DevelopSettings, MaskKind, MaskShape, NO_OBJECT_MODEL_SENTENCE, ObjectMask, mask_kinds_with,
};

/// How long the background worker waits before it starts the encoder, so that a drag of the Input's exposure (which asks
/// for new features on every tick) starts one encode, not a hundred.
pub(crate) const PREPARE_DEBOUNCE: Duration = Duration::from_millis(150);

/// How many planes a photo keeps (1.4 MB each at 1024 x 683).
const PLANES_KEPT: usize = 8;

/// How often a request that waits for the encoder looks at its own cancel flag.
const WAIT_STEP: Duration = Duration::from_millis(40);

/// The sentence for a mask index that is not an Object row of the photo.
const NOT_AN_OBJECT_ROW: &str =
    "That mask is not an Object mask. Select an Object mask and try again.";

/// The sentence for a click or a box whose numbers are not numbers.
const BAD_PROMPT: &str = "That click is not on the photo. Click the photo again.";

/// SAM's encoder output for a photo, with what it was made from.
pub(crate) struct ObjectFeatures {
    /// What the picture was made from ([`Renderer::object_key`]).
    pub(crate) key: String,
    /// FNV-1a of the key, in the keys of the planes.
    pub(crate) hash: u64,
    /// The encoder's three maps (about 16.8 MB).
    pub(crate) features: Features,
    /// The picture's size: its aspect is the photo frame's, and `S0` is cut to it.
    pub(crate) picture: (usize, usize),
    /// Milliseconds the picture and the encoder took (for the log and the tests).
    pub(crate) encode_ms: f64,
}

/// One run of the encoder.
struct Running {
    key: String,
    stop: CancelToken,
    /// Started in the background: leaving the photo or a newer wish stops it. A run a `mask_find` or an export
    /// asked for is never stopped by anyone but its caller.
    speculative: bool,
}

/// A wish of the background worker: the features of this picture, made from this photo.
struct Wanted {
    key: String,
    src: PhotoSource,
    /// When the wish was made: the worker starts once nobody has changed it for [`PREPARE_DEBOUNCE`].
    at: Instant,
}

#[derive(Default)]
struct SlotState {
    /// The latest features the photo has. A request for another picture replaces them when it finishes.
    ready: Option<Arc<ObjectFeatures>>,
    /// The encoder is running for this key.
    running: Option<Running>,
    /// What the background worker will make next.
    wanted: Option<Wanted>,
    /// A background worker is alive for the photo.
    worker: bool,
    /// The planes `S0` made lately, newest last.
    planes: VecDeque<Arc<SourcePlane>>,
    /// How many times the encoder finished for this photo (for the tests and the log).
    encodes: u64,
}

/// What a photo's graph keeps for its Object rows.
#[derive(Default)]
pub(crate) struct ObjectSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

impl ObjectSlot {
    /// Bytes the features and the planes hold.
    pub(crate) fn bytes(&self) -> usize {
        let st = lock(&self.state);
        st.ready.as_ref().map_or(0, |f| f.features.bytes())
            + st.planes.iter().map(|p| 2 * p.data.len()).sum::<usize>()
    }

    /// The person left the photo: nothing is wanted for it, and a run that was started in the background stops.
    pub(crate) fn abandon(&self) {
        let mut st = lock(&self.state);
        st.wanted = None;
        if let Some(run) = st.running.as_ref().filter(|r| r.speculative) {
            run.stop.cancel();
        }
    }

    /// Whether the encoder is working for the photo, or will be in a moment.
    pub(crate) fn preparing(&self) -> bool {
        let st = lock(&self.state);
        st.worker || st.running.is_some()
    }

    /// How many times the encoder finished for the photo.
    pub(crate) fn encodes(&self) -> u64 {
        lock(&self.state).encodes
    }

    /// The key of the features the photo has, if any.
    pub(crate) fn ready_key(&self) -> Option<String> {
        lock(&self.state).ready.as_ref().map(|f| f.key.clone())
    }

    fn plane(&self, key: u64) -> Option<Arc<SourcePlane>> {
        lock(&self.state)
            .planes
            .iter()
            .find(|p| p.key == key)
            .cloned()
    }

    fn keep_plane(&self, plane: Arc<SourcePlane>) {
        let mut st = lock(&self.state);
        if st.planes.iter().any(|p| p.key == plane.key) {
            return;
        }
        st.planes.push_back(plane);
        while st.planes.len() > PLANES_KEPT {
            st.planes.pop_front();
        }
    }
}

/// What a request remembers of its Object rows.
#[derive(Default)]
pub(crate) struct JobObjects {
    key: Option<String>,
    features: Option<Arc<ObjectFeatures>>,
    planes: Option<(Vec<u64>, Arc<ObjectPlanes>)>,
}

/// FNV-1a of a string.
pub(super) fn fnv(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The key of the plane of a prompt on features: equal keys are equal planes.
fn plane_key(features: u64, prompt: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in [features, prompt] {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn ml_error(e: MlError) -> RenderError {
    match e {
        MlError::Cancelled => RenderError::Cancelled,
        other => RenderError::Image(format!("The Object mask can't be made: {other}.")),
    }
}

// ---------------------------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------------------------

impl Renderer {
    /// The Object model this renderer has, when its weights are there and have not been found damaged.
    #[must_use]
    pub fn object_model(&self) -> Option<Arc<ObjectModel>> {
        self.options()
            .object_model
            .clone()
            .filter(|m| m.available())
    }

    /// Whether this renderer can make Object masks.
    #[must_use]
    pub fn object_available(&self) -> bool {
        self.object_model().is_some()
    }

    /// The kinds of mask this machine can make now (`docs/23` section 9, `mask_kinds`): Object when the model is there,
    /// Subject and People where the Vision probe said yes.
    #[must_use]
    pub fn mask_kinds(&self) -> Vec<MaskKind> {
        mask_kinds_with(self.mask_support())
    }

    /// The sentence for the first mask of the settings that this computer cannot make
    /// ([`unmakeable_mask_with`]), `None` when the photo can be rendered with all its masks.
    #[must_use]
    pub fn unmakeable_mask(&self, s: &DevelopSettings) -> Option<String> {
        unmakeable_mask_with(s, self.mask_support())
    }

    /// Who hears when the encoder starts or stops working for a photo ([`ObjectPhase`]). Replaces the one set before.
    pub fn set_object_listener(&self, f: super::ObjectListener) {
        *lock(&self.inner.object_listener) = Some(f);
    }

    fn object_notify(&self, id: PhotoId, phase: ObjectPhase) {
        let listener = lock(&self.inner.object_listener).clone();
        if let Some(f) = listener {
            f(id, phase);
        }
    }

    /// Whether the encoder is working for the photo, or will be in a moment ("Getting ready…").
    #[must_use]
    pub fn object_preparing(&self, id: PhotoId) -> bool {
        self.existing_graph(id)
            .is_some_and(|g| g.object.preparing())
    }

    /// How many times the encoder finished for the photo since its graph was made.
    #[must_use]
    pub fn object_encodes(&self, id: PhotoId) -> u64 {
        self.existing_graph(id).map_or(0, |g| g.object.encodes())
    }

    /// Whether the photo has features for the picture of these settings (an Object click would not wait).
    #[must_use]
    pub fn object_ready(&self, src: &PhotoSource) -> bool {
        let Ok(key) = self.object_key(src) else {
            return false;
        };
        self.existing_graph(src.id)
            .is_some_and(|g| g.object.ready_key().as_deref() == Some(key.as_str()))
    }

    /// The renderer that makes SAM's pictures: it has no graphics device, so the picture is the CPU reference path's
    /// bytes whatever the GPU policy (as Apple Auto's: [`AdvisorPictures`](crate::pipeline::AdvisorPictures)).
    pub(crate) fn object_reference(&self) -> Result<&Renderer, RenderError> {
        self.inner
            .object_reference
            .get_or_init(|| {
                let mut options = (*self.options()).clone();
                options.gpu = None;
                options.object_model = None;
                options.vision = None;
                Renderer::new(Arc::new(crate::pipeline::NeutralOnly), options)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// What SAM's picture of this photo is made from: the photo, the raw group's numbers and the Input's colour
    /// ([`object_source`]), and nothing a node, the look or the crop says. Equal keys are equal pictures.
    ///
    /// # Errors
    ///
    /// [`RenderError::Color`] for estimates that cannot make a picture.
    pub(crate) fn object_key(&self, src: &PhotoSource) -> Result<String, RenderError> {
        let reference = self.object_reference()?;
        let stripped = object_source(src);
        let cancel = CancelToken::new();
        let mut job = Job::new(reference, &stripped, &cancel)?;
        let raw = job.params()?.key();
        Ok(format!(
            "{:016x}|{raw:?}|{:?}",
            stripped.fingerprint,
            color_key(&stripped)
        ))
    }

    /// Makes the features of the photo's picture: the picture on the CPU reference path, then the encoder on the model's
    /// threads. `cancel` stops the picture and `also` (asked between the encoder's blocks) the encoder.
    fn make_object_features(
        &self,
        model: &ObjectModel,
        src: &PhotoSource,
        key: &str,
        cancel: &CancelToken,
        also: &(dyn Fn() -> bool + Sync),
    ) -> Result<ObjectFeatures, RenderError> {
        let started = Instant::now();
        let sam = model.sam()?;
        let loaded_ms = started.elapsed().as_secs_f64() * 1000.0;
        let picture = self.object_picture(src, fnv(key), cancel)?;
        let picture_ms = started.elapsed().as_secs_f64() * 1000.0 - loaded_ms;
        let (width, height) = (picture.width, picture.height);
        let pic = Picture::new(width, height, &picture.rgb).map_err(ml_error)?;
        let stopped = || cancel.is_cancelled() || also();
        let features = model
            .run(|| sam.encode_cancellable(&pic, &stopped))
            .map_err(ml_error)?;
        let encode_ms = started.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(
            ms = encode_ms,
            model_ms = loaded_ms,
            picture_ms,
            "the Object features (model, picture and encoder)"
        );
        Ok(ObjectFeatures {
            key: key.to_owned(),
            hash: fnv(key),
            features,
            picture: (width, height),
            encode_ms,
        })
    }

    /// The photo's features for `key`, made when it has none: waits for the encoder when it is running for this key, stops a
    /// run that was started in the background for another one, and runs the encoder itself when nobody does.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` is cancelled (or a run of this call was stopped because the person left the
    /// photo and the call is a background one), and the errors of the picture and the model.
    fn ensure_object_features(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &ObjectModel,
        src: &PhotoSource,
        key: &str,
        speculative: bool,
        cancel: &CancelToken,
    ) -> Result<Arc<ObjectFeatures>, RenderError> {
        let slot = &graph.object;
        loop {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            let mut st = lock(&slot.state);
            if let Some(ready) = st.ready.as_ref().filter(|f| f.key == key) {
                return Ok(Arc::clone(ready));
            }
            if let Some(run) = &st.running {
                // A run for another picture that nobody asked for in words is stopped; one a click is waiting for is not.
                if run.key != key && run.speculative && !speculative {
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
                key: key.to_owned(),
                stop: stop.clone(),
                speculative,
            });
            drop(st);
            let result =
                self.make_object_features(model, src, key, &stop, &|| cancel.is_cancelled());
            let mut st = lock(&slot.state);
            st.running = None;
            match result {
                Ok(features) => {
                    let features = Arc::new(features);
                    st.ready = Some(Arc::clone(&features));
                    st.encodes += 1;
                    slot.cv.notify_all();
                    return Ok(features);
                }
                Err(RenderError::Cancelled) => {
                    slot.cv.notify_all();
                    drop(st);
                    // Stopped by the caller, or by the person leaving the photo: a background call ends. A click that was
                    // waiting asks again.
                    if speculative || cancel.is_cancelled() {
                        return Err(RenderError::Cancelled);
                    }
                }
                Err(e) => {
                    slot.cv.notify_all();
                    return Err(e);
                }
            }
        }
    }

    /// Asks for the features of the photo's picture in the background (`docs/23` section 4.4: "as soon as an Object row exists
    /// on the photo"). Returns at once. Nothing is done when the model is missing, the photo has the features, or the encoder is
    /// working for them already; a worker waits until nobody has changed the wish for a moment (150 ms) so a stream of requests makes one encode.
    pub fn object_prepare(&self, src: &PhotoSource) {
        let Some(model) = self.object_model() else {
            return;
        };
        let Ok(key) = self.object_key(src) else {
            return;
        };
        let graph = self.graph(src.id);
        self.object_prepare_keyed(&graph, &model, src, key);
    }

    fn object_prepare_keyed(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &Arc<ObjectModel>,
        src: &PhotoSource,
        key: String,
    ) {
        {
            let mut st = lock(&graph.object.state);
            if st.ready.as_ref().is_some_and(|r| r.key == key)
                || st.running.as_ref().is_some_and(|r| r.key == key)
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
        }
        let renderer = self.clone();
        let (worker_graph, worker_model) = (Arc::clone(graph), Arc::clone(model));
        let spawned = std::thread::Builder::new()
            .name("yellowmoon-sam-prepare".to_owned())
            .spawn(move || renderer.object_worker(&worker_graph, &worker_model));
        if let Err(e) = spawned {
            tracing::error!(error = %e, "the Object worker could not start");
            let mut st = lock(&graph.object.state);
            st.worker = false;
            st.wanted = None;
        }
    }

    /// The background worker of a photo: makes what is wanted, one picture at a time, until nothing is. A wish is taken once it has
    /// been left alone for [`PREPARE_DEBOUNCE`], so a stream of them (a drag of the Input's exposure) makes one encode, of the last.
    fn object_worker(&self, graph: &Arc<PhotoGraph>, model: &Arc<ObjectModel>) {
        let id = graph.id();
        self.object_notify(id, ObjectPhase::Preparing);
        loop {
            let wanted = {
                let mut st = lock(&graph.object.state);
                match &st.wanted {
                    None => {
                        st.worker = false;
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
            match self.ensure_object_features(
                graph,
                model,
                &wanted.src,
                &wanted.key,
                true,
                &CancelToken::new(),
            ) {
                Ok(f) => {
                    tracing::debug!(ms = f.encode_ms, "the Object features are ready");
                }
                Err(RenderError::Cancelled) => {}
                Err(e) => tracing::warn!(error = %e, "the Object features could not be made"),
            }
        }
        self.object_notify(id, ObjectPhase::Idle);
    }

    /// The photo is no longer the one being looked at: the encoder stops working for it if it was started in the background.
    pub(crate) fn object_leave(&self, id: PhotoId) {
        if let Some(g) = self.existing_graph(id) {
            g.object.abandon();
        }
    }

    /// `mask_find` (`docs/23` section 9): adds the person's click or box to the Object row `mask` of `node`, runs the decoder
    /// and answers what the row would become, or that nothing was found. **It changes nothing**: the caller applies a
    /// found answer as one row patch. It waits for the encoder when it is running, and makes the features when the photo
    /// has none; the plane it made is kept, so the render that follows runs no decoder.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] with the plain sentence for a model that is missing, a mask that is not an Object row, a click that
    /// is not a number, and a row that has 256 points already (`An Object mask can have up to 256 points.`);
    /// [`RenderError::Cancelled`]; the errors of the picture.
    pub fn mask_find(
        &self,
        src: &PhotoSource,
        node: u32,
        mask: usize,
        prompt: ObjectPrompt,
        cancel: &CancelToken,
    ) -> Result<MaskFound, RenderError> {
        let model = self
            .object_model()
            .ok_or_else(|| RenderError::Mask(NO_OBJECT_MODEL_SENTENCE.to_owned()))?;
        let row: &ObjectMask = src
            .settings
            .nodes
            .corrector(node)
            .and_then(|c| c.masks.items.get(mask))
            .and_then(|r| match &r.shape {
                MaskShape::Object(o) => Some(o),
                _ => None,
            })
            .ok_or_else(|| RenderError::Mask(NOT_AN_OBJECT_ROW.to_owned()))?;
        let finite = match prompt {
            ObjectPrompt::Point { x, y, .. } => x.is_finite() && y.is_finite(),
            ObjectPrompt::Box { x0, y0, x1, y1 } => [x0, y0, x1, y1].iter().all(|v| v.is_finite()),
        };
        if !finite {
            return Err(RenderError::Mask(BAD_PROMPT.to_owned()));
        }
        let mut points = row.points.clone();
        let mut bbox = row.box_;
        match prompt {
            ObjectPrompt::Point { x, y, positive } => {
                if points.len() >= MAX_POINTS {
                    return Err(RenderError::Mask(
                        crate::settings::GraphError::TooManyObjectPoints.to_string(),
                    ));
                }
                points.push((round_coordinate(x), round_coordinate(y), positive));
            }
            ObjectPrompt::Box { x0, y0, x1, y1 } => {
                bbox = Some(round_box([x0, y0, x1, y1]));
            }
        }
        let key = self.object_key(src)?;
        let graph = self.graph(src.id);
        let features = self.ensure_object_features(&graph, &model, src, &key, false, cancel)?;
        let started = Instant::now();
        match decode_plane(&model, &graph, &features, &points, bbox)? {
            Ok(_) => Ok(MaskFound::Found {
                points,
                box_: bbox,
                took_ms: started.elapsed().as_secs_f64() * 1000.0,
            }),
            Err(nothing) => Ok(MaskFound::NothingFound(nothing)),
        }
    }

    /// The picture SAM looks at for this photo ([`object_source`]), as `(width, height, rgb)`, made again each time: for checks
    /// and diagnostics, like [`Renderer::blur_map`].
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn object_picture_of(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<(usize, usize, Vec<u8>), RenderError> {
        let p = self.object_picture_made(src, cancel)?;
        Ok((p.width, p.height, p.rgb))
    }

    /// `S0` of an Object row that has these points and this box (`docs/algorithms/06` section 6.3), the plane the colour pass
    /// resamples: the decoder's soft mask on the photo's features, made when the photo has none. `None` when the decoder
    /// finds nothing for the prompt. For checks and diagnostics, like [`Renderer::blur_map`].
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when there is no model; [`RenderError::Cancelled`]; the errors of the picture.
    pub fn object_plane(
        &self,
        src: &PhotoSource,
        points: &[(f64, f64, bool)],
        bbox: Option<[f64; 4]>,
        cancel: &CancelToken,
    ) -> Result<Option<Arc<SourcePlane>>, RenderError> {
        let model = self
            .object_model()
            .ok_or_else(|| RenderError::Mask(NO_OBJECT_MODEL_SENTENCE.to_owned()))?;
        let key = self.object_key(src)?;
        let graph = self.graph(src.id);
        let features = self.ensure_object_features(&graph, &model, src, &key, false, cancel)?;
        Ok(decode_plane(&model, &graph, &features, points, bbox)?.ok())
    }
}

/// The decoder on a prompt and the plane of its soft mask, kept in the photo's slot: `Ok(Ok(plane))` for a mask,
/// `Ok(Err(nothing))` when the decoder finds nothing there. The same plane (`plane_key`) is found again by the render that
/// follows, which runs no decoder.
fn decode_plane(
    model: &ObjectModel,
    graph: &PhotoGraph,
    features: &ObjectFeatures,
    points: &[(f64, f64, bool)],
    bbox: Option<[f64; 4]>,
) -> Result<Result<Arc<SourcePlane>, NothingFound>, RenderError> {
    let key = plane_key(features.hash, prompt_key(points, bbox));
    if let Some(plane) = graph.object.plane(key) {
        return Ok(Ok(plane));
    }
    let sam = model.sam()?;
    let decoding = sam
        .decode(&features.features, &sam_prompt(points, bbox))
        .map_err(ml_error)?;
    Ok(match decoding {
        Decoding::NothingFound {
            reason,
            object_score,
            area,
        } => Err(NothingFound {
            reason,
            object_score,
            area,
        }),
        Decoding::Found(found) => {
            let (w, h) = features.picture;
            let plane = Arc::new(plane_of(key, &found.soft_mask(w, h)));
            graph.object.keep_plane(Arc::clone(&plane));
            Ok(plane)
        }
    })
}

// ---------------------------------------------------------------------------------------------
// A request
// ---------------------------------------------------------------------------------------------

impl Job<'_> {
    /// The planes of the rows of the plan that ends at `stop` that read the Input's picture: for each enabled Object row that has a
    /// point or a box, `S0` of the decoder on its prompt (below), and for the Subject and People rows the union of Vision's
    /// instances (`vision::Job::vision_planes_for`). `None` when the plan has no such row, when this machine has no model, and (for a viewer) when
    /// the photo has no features yet: the encoder is started in the background and the rows select nothing until the
    /// listener tells the page to draw again.
    ///
    /// # Errors
    ///
    /// The errors of the picture and the model for an export, which makes the features itself; [`RenderError::Cancelled`].
    pub(crate) fn objects_for(
        &mut self,
        stop: PlanStop,
    ) -> Result<Option<Arc<ObjectPlanes>>, RenderError> {
        let objects = self.object_planes_for(stop)?;
        let vision = self.vision_planes_for(stop)?;
        if vision.is_empty() {
            return Ok(objects);
        }
        let mut merged = objects.map(|o| (*o).clone()).unwrap_or_default();
        for (kind, plane) in vision {
            merged.insert_vision(kind, plane);
        }
        Ok(Some(Arc::new(merged)))
    }

    /// The planes of the Object rows alone (see [`Job::objects_for`]).
    fn object_planes_for(
        &mut self,
        stop: PlanStop,
    ) -> Result<Option<Arc<ObjectPlanes>>, RenderError> {
        let rows: Vec<ObjectMask> = object_rows(&self.src.settings, stop)
            .into_iter()
            .filter(|o| asks_for_mask(o))
            .cloned()
            .collect();
        if rows.is_empty() {
            return Ok(None);
        }
        let Some(model) = self.opts.object_model.clone().filter(|m| m.available()) else {
            return Ok(None);
        };
        let mut prompts: Vec<u64> = rows.iter().map(|o| prompt_key(&o.points, o.box_)).collect();
        prompts.sort_unstable();
        prompts.dedup();
        if let Some((known, planes)) = &self.object.planes
            && *known == prompts
        {
            return Ok(Some(Arc::clone(planes)));
        }
        let Some(features) = self.object_features(&model)? else {
            return Ok(None);
        };
        let mut planes = ObjectPlanes::default();
        for o in &rows {
            self.check()?;
            // A prompt the decoder refuses (a hand-written file, other features) selects nothing.
            if let Ok(plane) = decode_plane(&model, &self.g, &features, &o.points, o.box_)? {
                planes.insert(prompt_key(&o.points, o.box_), plane);
            }
        }
        let planes = Arc::new(planes);
        self.object.planes = Some((prompts, Arc::clone(&planes)));
        Ok(Some(planes))
    }

    /// The features for this request's photo: the photo's own when they are for the same picture; else, for an export, made now
    /// for this request alone; for a viewer, the latest ones the photo has (or none), after asking for the right ones.
    fn object_features(
        &mut self,
        model: &Arc<ObjectModel>,
    ) -> Result<Option<Arc<ObjectFeatures>>, RenderError> {
        if let Some(f) = &self.object.features {
            return Ok(Some(Arc::clone(f)));
        }
        let key = match &self.object.key {
            Some(k) => k.clone(),
            None => {
                let k = self.r.object_key(self.src)?;
                self.object.key = Some(k.clone());
                k
            }
        };
        let ready = lock(&self.g.object.state).ready.clone();
        if let Some(f) = ready.as_ref().filter(|f| f.key == key) {
            self.object.features = Some(Arc::clone(f));
            return Ok(Some(Arc::clone(f)));
        }
        if self.restartable {
            self.check()?;
            let cancel = self.cancel;
            let made = self
                .r
                .make_object_features(model, self.src, &key, cancel, &|| false)?;
            let made = Arc::new(made);
            self.object.features = Some(Arc::clone(&made));
            return Ok(Some(made));
        }
        self.r.object_prepare_keyed(&self.g, model, self.src, key);
        Ok(ready)
    }
}

/// Plans whose Object rows ask for a mask, as the cache key of a grade plan names them: the plane's key, or `none` for a row
/// whose features are not there (it selects nothing and says so).
#[must_use]
pub(crate) fn plane_text(planes: Option<&Arc<ObjectPlanes>>, o: &ObjectMask) -> String {
    planes
        .and_then(|p| p.plane_for(o))
        .map_or_else(|| "none".to_owned(), |p| format!("{:x}", p.key))
}

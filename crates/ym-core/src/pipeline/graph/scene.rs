//! The element masks of Reference Grade in the render graph (`docs/34-reference-grade.md` sections 4a.3, 4b and 7.1, RG-04;
//! `docs/23-nodes.md` section 4.5 for the machinery it follows).
//!
//! A Reference Grade node's element grades act through **soft masks** that `ym_color::mask_element` makes from the photo's
//! **scene**: for every pixel of the Input's picture, which of the seven elements it belongs to (`ym_ml::scene`: the segmenter's
//! label map, MediaPipe's skin inside `person`, the subject rule). The picture is the one Object, Subject, People and Depth read
//! ([`object_source`](crate::pipeline::object::object_source) rendered by [`Renderer::object_picture`]: the photo frame, the
//! Input's colour, the Neutral look, 8 bits, sRGB, 1024 px on the long edge, no output sharpening). There is no second
//! picture path.
//!
//! * **The scene** (1024 x 683 bytes for a 3:2 photo, 0.7 MB) lives in the photo's graph ([`SceneSlot`]), in memory, and never
//!   touches the disk. Its key names what it was made from ([`Renderer::scene_key`]): the picture's key (the raw group's numbers and
//!   the Input's colour, never a node's grade, the look, the crop, the turn or the lens character), the weights and the rules'
//!   version. Grading a node, changing the look or moving a slider of another node never makes a scene again; the Input's exposure
//!   does, as it moves the picture. It is dropped with the photo's graph.
//! * **The soft masks are made at the size they are needed** (`ym_color::mask_element`): per render window, from the scene and the
//!   guide, so that a fit, the viewer and an export each get their own size from the one scene, and only for the elements whose
//!   effective grade is not the identity ([`live_element_nodes`]).
//! * **A viewer never waits.** A frame takes the scene the photo has for its picture and, for another picture's scene (the Input's
//!   exposure moved), keeps the latest one meanwhile. For a photo with none yet it asks the worker for one in the background
//!   ([`Renderer::scene_prepare`]: one worker per photo that waits [`PREPARE_DEBOUNCE`], so a drag of the Input's exposure makes one
//!   scene, not a hundred) and the element step does nothing until the listener ([`ScenePhase`]) tells the page to draw again. The
//!   app says "Finding…" meanwhile (`viewport_busy.finding`). A scene that failed is kept as a failure of its key and a viewer does
//!   not ask again; the frame says why.
//! * **An export makes its own scene** when the photo has none of the right key, on the CPU, so its bytes depend on nothing but the
//!   file and the settings, and `ym` and the app agree. **A copy of Yellowmoon with no scene model** draws the viewer's frame
//!   without the element step and says so, and refuses an export with the `mask` error ([`Renderer::unmakeable_reference_node`]).

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ym_color::mask_element::ElementScene;
use ym_color::plan::Stop as PlanStop;
use ym_gpu::queue::CancelToken;
use ym_ml::scene::RULES_VERSION;

use super::object::fnv;
use super::{Job, PhotoGraph, Renderer, lock};
use crate::pipeline::RenderError;
use crate::pipeline::color::{PhotoSource, live_element_nodes};
use crate::pipeline::scene::{
    NO_SCENE_MODEL_SENTENCE, NO_SKIN_MODEL_SENTENCE, SCENE_FORMAT_VERSION,
    SCENE_MODEL_MISSING_NOTE, SKIN_MODEL_MISSING_NOTE, SceneModel, ScenePhase,
};
use crate::reference::ReferencePicture;
use crate::session::PhotoId;
use crate::settings::DevelopSettings;

/// How long the background worker waits before it starts the networks, so that a drag of the Input's exposure (which asks for a
/// new scene on every tick) makes one scene, not a hundred (the same wait as the Object encoder's).
pub(crate) const PREPARE_DEBOUNCE: Duration = super::object::PREPARE_DEBOUNCE;

/// How often a request that waits for the worker looks at its own cancel flag.
const WAIT_STEP: Duration = Duration::from_millis(40);

/// Who hears when the scene worker starts or ends its work for a photo ([`Renderer::set_scene_listener`]).
pub(crate) type SceneListener = Arc<dyn Fn(PhotoId, ScenePhase) + Send + Sync>;

/// What a scene is made from: the text names the Input's picture, the weights and the rules; the hash names the picture alone (the
/// key of the picture Object shares).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneKey {
    /// The scene's key: the picture's key (the one Object reads), the weights and the rules' version.
    pub text: String,
    /// The FNV-1a of the picture's key: the picture the workers share is kept under it.
    picture: u64,
}

/// The scene of a reference picture ([`Renderer::reference_scene`]).
#[derive(Debug, Clone)]
pub struct ReferenceScene {
    /// The seven element bits at every pixel of the reference's scene picture.
    pub scene: Arc<ElementScene>,
    /// Whether skin could be looked for (the Selfie weights were there).
    pub skin_available: bool,
}

/// The photo's scene with what it was made from.
pub(crate) struct SceneEntry {
    /// What the scene was made from.
    pub(crate) key: String,
    /// The scene.
    pub(crate) scene: Arc<ElementScene>,
    /// Whether skin could be looked for (the Selfie weights were there).
    pub(crate) skin_available: bool,
    /// Milliseconds the picture and the networks took (for the log and the tests).
    pub(crate) ms: f64,
}

/// One run of the networks.
struct Running {
    key: String,
    stop: CancelToken,
    /// Started in the background: leaving the photo or a newer wish stops it. A run an export asked for is never stopped by anyone
    /// but its caller.
    speculative: bool,
}

/// A wish of the background worker: the scene of this picture, made from this photo.
struct Wanted {
    key: SceneKey,
    src: PhotoSource,
    /// When the wish was made: the worker starts once nobody has changed it for [`PREPARE_DEBOUNCE`].
    at: Instant,
}

#[derive(Default)]
struct SlotState {
    /// The latest scene the photo has. A request for another picture replaces it when it finishes.
    ready: Option<Arc<SceneEntry>>,
    /// The scene of this key could not be made, and why (the sentence). Nothing asks again until the key changes.
    failed: Option<(String, String)>,
    /// The networks are running for this key.
    running: Option<Running>,
    /// What the background worker will make next.
    wanted: Option<Wanted>,
    /// A background worker is alive for the photo (it may be waiting out the debounce).
    worker: bool,
    /// The worker is making a scene (the picture or the networks are running): the nodes say "Finding…".
    finding: bool,
    /// How many times the networks finished for the photo, workers and exports together.
    makes: u64,
    /// How many workers were started for the photo.
    workers: u64,
}

/// What a photo's graph keeps for its Reference Grade nodes' elements.
#[derive(Default)]
pub(crate) struct SceneSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

impl SceneSlot {
    /// Bytes the scene holds (two bytes a pixel: the element bits and the segmenter's element number).
    pub(crate) fn bytes(&self) -> usize {
        lock(&self.state)
            .ready
            .as_ref()
            .map_or(0, |e| e.scene.bits.len() + e.scene.classes.len())
    }

    /// The person left the photo: nothing is wanted for it, and a run that was started in the background stops.
    pub(crate) fn abandon(&self) {
        let mut st = lock(&self.state);
        st.wanted = None;
        if let Some(run) = st.running.as_ref().filter(|r| r.speculative) {
            run.stop.cancel();
        }
    }

    /// Whether the worker is making the scene for the photo ("Finding…").
    pub(crate) fn working(&self) -> bool {
        lock(&self.state).finding
    }

    /// Whether a worker thread is alive for the photo, even one that only waits out the debounce.
    pub(crate) fn worker_alive(&self) -> bool {
        lock(&self.state).worker
    }

    /// How many times the networks finished for the photo.
    pub(crate) fn makes(&self) -> u64 {
        lock(&self.state).makes
    }

    /// How many workers were started for the photo.
    pub(crate) fn workers(&self) -> u64 {
        lock(&self.state).workers
    }

    /// The scene the photo has, whatever picture it was made from.
    pub(crate) fn latest(&self) -> Option<Arc<SceneEntry>> {
        lock(&self.state).ready.clone()
    }

    /// The sentence of the failure kept for this key, if the photo has one.
    pub(crate) fn failure(&self, key: &str) -> Option<String> {
        lock(&self.state)
            .failed
            .as_ref()
            .filter(|(k, _)| k == key)
            .map(|(_, why)| why.clone())
    }

    /// The key of the scene the photo has, if any.
    pub(crate) fn ready_key(&self) -> Option<String> {
        lock(&self.state).ready.as_ref().map(|e| e.key.clone())
    }
}

/// What a request remembers of its Reference Grade nodes' scene.
#[derive(Default)]
pub(crate) struct JobScene {
    key: Option<SceneKey>,
    scene: Option<Arc<ElementScene>>,
}

// ---------------------------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------------------------

impl Renderer {
    /// The scene model this renderer has, when its weights are there and have not been found damaged.
    #[must_use]
    pub fn scene_model(&self) -> Option<Arc<SceneModel>> {
        self.options().scene_model.clone().filter(|m| m.available())
    }

    /// Whether this renderer can make element masks.
    #[must_use]
    pub fn scene_available(&self) -> bool {
        self.scene_model().is_some()
    }

    /// Who hears when the scene worker starts or ends its work for a photo ([`ScenePhase`]). Replaces the one set before.
    pub fn set_scene_listener(&self, f: SceneListener) {
        *lock(&self.inner.scene_listener) = Some(f);
    }

    fn scene_notify(&self, id: PhotoId, phase: ScenePhase) {
        let listener = lock(&self.inner.scene_listener).clone();
        if let Some(f) = listener {
            f(id, phase);
        }
    }

    /// Whether the worker is making the scene for the photo ("Finding…").
    #[must_use]
    pub fn scene_working(&self, id: PhotoId) -> bool {
        self.existing_graph(id).is_some_and(|g| g.scene.working())
    }

    /// Whether a background worker thread is alive for the photo, including one that only waits out the debounce.
    #[must_use]
    pub fn scene_worker_alive(&self, id: PhotoId) -> bool {
        self.existing_graph(id)
            .is_some_and(|g| g.scene.worker_alive())
    }

    /// How many times the networks finished for the photo since its graph was made.
    #[must_use]
    pub fn scene_makes(&self, id: PhotoId) -> u64 {
        self.existing_graph(id).map_or(0, |g| g.scene.makes())
    }

    /// How many background workers were started for the photo since its graph was made.
    #[must_use]
    pub fn scene_workers(&self, id: PhotoId) -> u64 {
        self.existing_graph(id).map_or(0, |g| g.scene.workers())
    }

    /// Whether the photo has a scene for the picture of these settings (a render would not wait).
    #[must_use]
    pub fn scene_ready(&self, src: &PhotoSource) -> bool {
        let Ok(key) = self.scene_key(src) else {
            return false;
        };
        self.existing_graph(src.id)
            .is_some_and(|g| g.scene.ready_key().as_deref() == Some(key.text.as_str()))
    }

    /// The sentence of the failure the photo keeps for the picture of these settings ("The Elements can't be made: …"), when its
    /// scene could not be made.
    #[must_use]
    pub fn scene_failure(&self, src: &PhotoSource) -> Option<String> {
        let key = self.scene_key(src).ok()?;
        self.existing_graph(src.id)?.scene.failure(&key.text)
    }

    /// What the photo's scene is made from: the picture (the one Object reads: the photo, the raw group's numbers and the Input's
    /// colour, and nothing a node, the look, the crop, the turn or the lens character says), the weights and the rules. Equal keys
    /// are equal scenes.
    ///
    /// # Errors
    ///
    /// [`RenderError::Color`] for estimates that cannot make a picture, and [`RenderError::Mask`] with
    /// [`NO_SCENE_MODEL_SENTENCE`] when the renderer has no model.
    pub fn scene_key(&self, src: &PhotoSource) -> Result<SceneKey, RenderError> {
        let model = self
            .scene_model()
            .ok_or_else(|| RenderError::Mask(NO_SCENE_MODEL_SENTENCE.to_owned()))?;
        let picture_key = self.object_key(src)?;
        Ok(SceneKey {
            text: format!(
                "{picture_key}|scene|{}|f{SCENE_FORMAT_VERSION}|r{RULES_VERSION}",
                model.weights_id()
            ),
            picture: fnv(&picture_key),
        })
    }

    /// Makes the photo's scene for `key`: the picture on the CPU reference path (the one Object reads, kept), then the networks on
    /// the model's threads. `cancel` stops the picture and `also` (asked between the networks' blocks) the networks.
    fn make_scene_entry(
        &self,
        model: &SceneModel,
        src: &PhotoSource,
        key: &SceneKey,
        cancel: &CancelToken,
        also: &(dyn Fn() -> bool + Sync),
    ) -> Result<SceneEntry, RenderError> {
        let started = Instant::now();
        let picture = self.object_picture(src, key.picture, cancel)?;
        let picture_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stopped = || cancel.is_cancelled() || also();
        let made = model.scene_of(&picture.rgb, picture.width, picture.height, &stopped)?;
        if cancel.is_cancelled() {
            return Err(RenderError::Cancelled);
        }
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(ms, picture_ms, "the scene (the picture and the networks)");
        Ok(SceneEntry {
            key: key.text.clone(),
            skin_available: made.skin_available,
            scene: Arc::new(ElementScene {
                key: fnv(&key.text),
                width: made.width,
                height: made.height,
                bits: made.bits,
                classes: made.classes,
            }),
            ms,
        })
    }

    /// The photo's scene for `key`, made when it has none: waits for the networks when they are running for this key, stops a run
    /// that was started in the background for another one, and runs the networks itself when nobody does. A scene that cannot be
    /// made is kept as the failure of its key.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` is cancelled (or a run of this call was stopped because the person left the photo
    /// and the call is a background one), and the errors of the picture and the model.
    fn ensure_scene_entry(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &SceneModel,
        src: &PhotoSource,
        key: &SceneKey,
        speculative: bool,
        cancel: &CancelToken,
    ) -> Result<Arc<SceneEntry>, RenderError> {
        let slot = &graph.scene;
        loop {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            let mut st = lock(&slot.state);
            if let Some(ready) = st.ready.as_ref().filter(|e| e.key == key.text) {
                return Ok(Arc::clone(ready));
            }
            if let Some(run) = &st.running {
                // A run for another picture that nobody asked for in words is stopped; one an export is waiting for is not.
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
            let result = self.make_scene_entry(model, src, key, &stop, &|| cancel.is_cancelled());
            let mut st = lock(&slot.state);
            st.running = None;
            match result {
                Ok(entry) => {
                    let entry = Arc::new(entry);
                    st.ready = Some(Arc::clone(&entry));
                    st.failed = None;
                    st.makes += 1;
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

    /// Asks for the scene of the photo's picture in the background (`docs/34` section 4a.3, step 4: "made per photo, in the
    /// background"). Returns at once. Nothing is done when the model is missing, the photo has the scene, the networks are working
    /// for it, it failed for this picture, or a worker already wants it; a worker waits until nobody has changed the wish for a
    /// moment (150 ms) so a stream of requests makes one scene.
    pub fn scene_prepare(&self, src: &PhotoSource) {
        let Some(model) = self.scene_model() else {
            return;
        };
        let Ok(key) = self.scene_key(src) else {
            return;
        };
        let graph = self.graph(src.id);
        self.scene_prepare_keyed(&graph, &model, src, key);
    }

    fn scene_prepare_keyed(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &Arc<SceneModel>,
        src: &PhotoSource,
        key: SceneKey,
    ) {
        {
            let mut st = lock(&graph.scene.state);
            if st.ready.as_ref().is_some_and(|r| r.key == key.text)
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
            .name("yellowmoon-scene-prepare".to_owned())
            .spawn(move || renderer.scene_worker(&worker_graph, &worker_model));
        if let Err(e) = spawned {
            tracing::error!(error = %e, "the scene worker could not start");
            let mut st = lock(&graph.scene.state);
            st.worker = false;
            st.wanted = None;
        }
    }

    /// The background worker of a photo: makes what is wanted, one picture at a time, until nothing is. A wish is taken once it
    /// has been left alone for [`PREPARE_DEBOUNCE`], so a stream of them (a drag of the Input's exposure) makes one scene, of the
    /// last picture. The listener hears [`ScenePhase::Started`] when the first run begins and [`ScenePhase::Finished`] when the
    /// worker ends; a worker whose wishes were all dropped before a run began (the person left the photo) ends without a word.
    fn scene_worker(&self, graph: &Arc<PhotoGraph>, model: &Arc<SceneModel>) {
        let id = graph.id();
        let mut announced = false;
        let mut failed: Option<String> = None;
        loop {
            let wanted = {
                let mut st = lock(&graph.scene.state);
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
            lock(&graph.scene.state).finding = true;
            if !announced {
                announced = true;
                self.scene_notify(id, ScenePhase::Started);
            }
            match self.ensure_scene_entry(
                graph,
                model,
                &wanted.src,
                &wanted.key,
                true,
                &CancelToken::new(),
            ) {
                Ok(e) => {
                    failed = None;
                    tracing::debug!(ms = e.ms, "the scene is ready");
                }
                Err(RenderError::Cancelled) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "the scene could not be made");
                    failed = Some(e.to_string());
                }
            }
        }
        if announced {
            self.scene_notify(id, ScenePhase::Finished { failed });
        }
    }

    /// The photo is no longer the one being looked at: the worker stops working for it if it started on its own.
    pub(crate) fn scene_leave(&self, id: PhotoId) {
        if let Some(g) = self.existing_graph(id) {
            g.scene.abandon();
        }
    }

    /// The scene of the photo for the picture of these settings, made now when the photo has none (waiting for the worker when it
    /// runs): the call of the fit (RG-06) and of checks and diagnostics. The soft masks of its elements at any size are
    /// `ym_color::mask_element::soft_masks`.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when there is no model or the scene cannot be made; [`RenderError::Cancelled`]; the errors of the
    /// picture.
    pub fn scene_of(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Arc<ElementScene>, RenderError> {
        let model = self
            .scene_model()
            .ok_or_else(|| RenderError::Mask(NO_SCENE_MODEL_SENTENCE.to_owned()))?;
        let key = self.scene_key(src)?;
        let graph = self.graph(src.id);
        self.ensure_scene_entry(&graph, &model, src, &key, false, cancel)
            .map(|e| Arc::clone(&e.scene))
    }

    /// The renderer that makes a strip photo's snapshot (`docs/34` section 6.4): this renderer's options without the graphics
    /// device, the segmenter, SAM and Vision, and this renderer's looks, so the picture is made on the CPU reference path, the same
    /// bytes on every machine, through the photo's own look.
    pub(crate) fn snapshot_renderer(&self) -> Result<&Renderer, RenderError> {
        self.inner
            .snapshot_reference
            .get_or_init(|| {
                let mut options = (*self.options()).clone();
                options.gpu = None;
                options.object_model = None;
                options.vision = None;
                options.scene_model = None;
                Renderer::new(Arc::clone(&self.inner.looks), options)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The scene of a **reference** picture (`docs/34` section 6, RG-05): the same segmenter, MediaPipe's skin and rules the photo's
    /// scene uses (`ym_ml::scene`), on the reference's scene picture ([`ReferencePicture::scene`], at most 1024 px on the long
    /// edge, as sRGB display codes). The soft masks of its elements at the analysis picture's size are
    /// `ym_color::mask_element::soft_masks_of_picture` with the guide of the analysis picture (`docs/34` decision 44).
    ///
    /// Made again each call: the fit (RG-06) keeps the result for the session, and the key names the picture and the weights, so an
    /// equal reference and an equal model are an equal scene.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when there is no model or the scene cannot be made; [`RenderError::Cancelled`].
    pub fn reference_scene(
        &self,
        reference: &ReferencePicture,
        cancel: &CancelToken,
    ) -> Result<ReferenceScene, RenderError> {
        let model = self
            .scene_model()
            .ok_or_else(|| RenderError::Mask(NO_SCENE_MODEL_SENTENCE.to_owned()))?;
        let picture = reference.scene();
        let rgb = picture.srgb8();
        let made = model.scene_of(&rgb, picture.width, picture.height, &|| {
            cancel.is_cancelled()
        })?;
        if cancel.is_cancelled() {
            return Err(RenderError::Cancelled);
        }
        let key = format!(
            "reference:{:016x}|scene|{}|f{SCENE_FORMAT_VERSION}|r{RULES_VERSION}",
            reference.digest(),
            model.weights_id()
        );
        Ok(ReferenceScene {
            skin_available: made.skin_available,
            scene: Arc::new(ElementScene {
                key: fnv(&key),
                width: made.width,
                height: made.height,
                bits: made.bits,
                classes: made.classes,
            }),
        })
    }

    /// The sentence for a Reference Grade node that has a live element grade in a photo of this computer when it has no scene
    /// model (`docs/34` decision 26: an export does not write the photo without the element step, as it does not for a Depth row
    /// without its model), or a live Skin grade and no skin model; `None` otherwise. A bypassed node and one whose element
    /// grades are all at the identity (Strength 0 included) need no model.
    #[must_use]
    pub fn unmakeable_reference_node(&self, s: &DevelopSettings) -> Option<String> {
        let live = crate::pipeline::color::live_element_nodes(s, PlanStop::Output);
        if live.is_empty() {
            return None;
        }
        let Some(model) = self.scene_model() else {
            return Some(NO_SCENE_MODEL_SENTENCE.to_owned());
        };
        (live.iter().any(|(_, l)| l[5]) && !model.skin_available())
            .then(|| NO_SKIN_MODEL_SENTENCE.to_owned())
    }
}

// ---------------------------------------------------------------------------------------------
// A request
// ---------------------------------------------------------------------------------------------

impl Job<'_> {
    /// The photo's scene for the plan that ends at `stop`, when that plan has a Reference Grade node with a live element grade.
    /// `None` for a plan with none, when this machine has no model (a viewer says so in the frame's notes and an export fails with
    /// the `mask` error), and (for a viewer) when the photo has no scene yet: the worker is started in the background and the
    /// element step does nothing until the listener tells the page to draw again. A viewer keeps the latest scene of another
    /// picture meanwhile. An export makes the scene itself when the photo has none of the right key.
    ///
    /// # Errors
    ///
    /// For an export, [`RenderError::Mask`] with the plain sentence; [`RenderError::Cancelled`]; the errors of the picture.
    pub(crate) fn scene_for(
        &mut self,
        settings: &DevelopSettings,
        stop: PlanStop,
    ) -> Result<Option<Arc<ElementScene>>, RenderError> {
        let live = live_element_nodes(settings, stop);
        if live.is_empty() {
            return Ok(None);
        }
        let Some(model) = self.opts.scene_model.clone().filter(|m| m.available()) else {
            if self.restartable {
                return Err(RenderError::Mask(NO_SCENE_MODEL_SENTENCE.to_owned()));
            }
            self.note(SCENE_MODEL_MISSING_NOTE);
            return Ok(None);
        };
        let wants_skin = live.iter().any(|(_, l)| l[5]);
        if wants_skin && !model.skin_available() {
            if self.restartable {
                return Err(RenderError::Mask(NO_SKIN_MODEL_SENTENCE.to_owned()));
            }
            self.note(SKIN_MODEL_MISSING_NOTE);
        }
        if let Some(scene) = &self.scene.scene {
            return Ok(Some(Arc::clone(scene)));
        }
        let key = match &self.scene.key {
            Some(k) => k.clone(),
            None => {
                let k = self.r.scene_key(self.src)?;
                self.scene.key = Some(k.clone());
                k
            }
        };
        let ready = self.g.scene.latest();
        if let Some(e) = ready.as_ref().filter(|e| e.key == key.text) {
            self.skin_made(wants_skin, e.skin_available)?;
            self.scene.scene = Some(Arc::clone(&e.scene));
            return Ok(Some(Arc::clone(&e.scene)));
        }
        if self.restartable {
            self.check()?;
            let cancel = self.cancel;
            let entry = self
                .r
                .ensure_scene_entry(&self.g, &model, self.src, &key, false, cancel)?;
            self.skin_made(wants_skin, entry.skin_available)?;
            self.scene.scene = Some(Arc::clone(&entry.scene));
            return Ok(Some(Arc::clone(&entry.scene)));
        }
        // A scene that failed for this picture is not asked for again; the frame says why and the step does nothing.
        match self.g.scene.failure(&key.text) {
            Some(why) => self.note(&why),
            None => self.r.scene_prepare_keyed(&self.g, &model, self.src, key),
        }
        // Meanwhile the latest scene, from another picture: it stays put while the picture moves.
        Ok(ready.map(|e| Arc::clone(&e.scene)))
    }

    /// A scene was made with no skin although a node asks for it (the skin network failed to load when it ran): an export fails
    /// with the `mask` error, a viewer says so.
    fn skin_made(&mut self, wanted: bool, made: bool) -> Result<(), RenderError> {
        if !wanted || made {
            return Ok(());
        }
        if self.restartable {
            return Err(RenderError::Mask(NO_SKIN_MODEL_SENTENCE.to_owned()));
        }
        self.note(SKIN_MODEL_MISSING_NOTE);
        Ok(())
    }
}

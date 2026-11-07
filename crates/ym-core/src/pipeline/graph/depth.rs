//! The Depth mask in the render graph (`docs/23-nodes.md` sections 3.10a and 4.8, `docs/algorithms/06-masks.md`
//! sections 5.8 and 8).
//!
//! A Depth row's raster `S0` is its band cut from the photo's **depth map** (`ym_color::mask_depth`): the relative inverse
//! depth `ym_ml::depth` makes of the Input's picture ([`object_source`](crate::pipeline::object::object_source): the
//! photo frame, the Input's colour, the Neutral look, 8 bits, sRGB, 1024 px on the long edge, no output sharpening),
//! rendered by the renderer of its own that has no graphics device, so the picture, the map and the mask are the CPU
//! reference path's bytes whatever the machine, and the app and `ym` agree. It is **the picture Object reads**: a photo
//! with an Object row and a Depth row makes it once ([`Renderer::object_picture`]).
//!
//! * **The map** (about 3.2 MB) lives in the photo's graph ([`DepthSlot`]), in memory, and never touches the disk. Its key
//!   names what the picture was made from ([`Renderer::depth_key`]): the raw group's numbers and the Input's colour (never a
//!   node's grade, the look, the crop, the turn or the lens character), the weights and the rank rules' version. A
//!   change of `near`, `far` or `softness` cuts a new plane from the cached ranks and never runs the network.
//! * **The worker** ([`Renderer::depth_prepare`]) makes the map in the background as soon as the core sees a Depth row on
//!   the photo the person is looking at: one worker per photo that waits [`PREPARE_DEBOUNCE`] (a drag of the Input's
//!   exposure asks for a new map on every tick), runs the picture and the network, and tells the listener
//!   ([`DepthPhase`]). A run started in the background stops when the person leaves the photo or asks for another
//!   picture's map (the network checks between its blocks and stages).
//! * **A viewer never waits.** A frame takes the map the photo has for its picture; for another picture's map (the
//!   Input's exposure moved) it keeps the latest one meanwhile, and a photo with none selects nothing until the listener
//!   tells the page to draw again. A map that failed is kept as a failure for its key and nothing asks again until the
//!   key changes.
//! * **An export makes its own map** when the photo has none of the right key, on the CPU, and does not keep it, so its
//!   bytes depend on nothing but the file and the settings. A map that cannot be made is the `mask` error.
//! * **`sample_depth` and `focus_plane`** read the cached map (waiting for the worker when it runs) and run no network of
//!   their own beyond making a map nobody has asked for.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ym_color::mask_depth::{
    DEPTH_RULES_VERSION, DepthMap, FOCUS_BOX_DEFAULT, pick_band, pick_depth,
};
use ym_color::plan::Stop as PlanStop;
use ym_gpu::queue::CancelToken;

use super::lens::FocusFrame;
use super::object::fnv;
use super::{Job, PhotoGraph, Renderer, lock};
use crate::pipeline::RenderError;
use crate::pipeline::color::{PhotoSource, uses_depth};
use crate::pipeline::depth::{
    DEPTH_FORMAT_VERSION, DepthModel, DepthPhase, FocusPlane, FocusReason,
};
use crate::pipeline::geometry::sensor_to_oriented;
use crate::pipeline::object::object_source;
use crate::session::PhotoId;
use crate::settings::{DevelopSettings, NO_DEPTH_MODEL_SENTENCE};

/// How long the background worker waits before it starts the network, so that a drag of the Input's exposure (which asks
/// for a new map on every tick) makes one map, not a hundred (the same wait as the Object encoder's).
pub(crate) const PREPARE_DEBOUNCE: Duration = super::object::PREPARE_DEBOUNCE;

/// How often a request that waits for the worker looks at its own cancel flag.
const WAIT_STEP: Duration = Duration::from_millis(40);

/// The sentence for a click that is not a place on the photo.
const BAD_POINT: &str = "That click is not on the photo. Click the photo again.";

/// Who hears when the depth worker starts or ends its work for a photo ([`Renderer::set_depth_listener`]).
pub(crate) type DepthListener = Arc<dyn Fn(PhotoId, DepthPhase) + Send + Sync>;

/// What a depth map is made from: the text names the Input's picture, the weights and the rules; the hash names the
/// picture alone (the key of the picture Object shares).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthKey {
    /// The map's key: the picture's key (the one Object reads), the weights and the rules' version.
    pub text: String,
    /// The FNV-1a of the picture's key: the picture Object and Depth share is kept under it.
    picture: u64,
}

/// The photo's depth map with what it was made from.
pub(crate) struct DepthEntry {
    /// What the map was made from.
    pub(crate) key: String,
    /// The map.
    pub(crate) map: Arc<DepthMap>,
    /// Milliseconds the picture and the network took (for the log and the tests).
    pub(crate) ms: f64,
}

/// One run of the network.
struct Running {
    key: String,
    stop: CancelToken,
    /// Started in the background: leaving the photo or a newer wish stops it. A run a click or an export asked for is
    /// never stopped by anyone but its caller.
    speculative: bool,
}

/// A wish of the background worker: the map of this picture, made from this photo.
struct Wanted {
    key: DepthKey,
    src: PhotoSource,
    /// When the wish was made: the worker starts once nobody has changed it for [`PREPARE_DEBOUNCE`].
    at: Instant,
}

#[derive(Default)]
struct SlotState {
    /// The latest map the photo has. A request for another picture replaces it when it finishes.
    ready: Option<Arc<DepthEntry>>,
    /// The map of this key could not be made, and why (the sentence). Nothing asks again until the key changes.
    failed: Option<(String, String)>,
    /// The network is running for this key.
    running: Option<Running>,
    /// What the background worker will make next.
    wanted: Option<Wanted>,
    /// A background worker is alive for the photo (it may be waiting out the debounce).
    worker: bool,
    /// The worker is making a map (the picture or the network is running): the rows say "Finding…".
    finding: bool,
    /// How many times the network finished for the photo, workers, clicks and the like together.
    networks: u64,
    /// How many workers were started for the photo.
    workers: u64,
}

/// What a photo's graph keeps for its Depth rows.
#[derive(Default)]
pub(crate) struct DepthSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

impl DepthSlot {
    /// Bytes the map holds (`d` and the ranks, 8 bytes a cell).
    pub(crate) fn bytes(&self) -> usize {
        lock(&self.state)
            .ready
            .as_ref()
            .map_or(0, |e| 8 * e.map.len())
    }

    /// The person left the photo: nothing is wanted for it, and a run that was started in the background stops.
    pub(crate) fn abandon(&self) {
        let mut st = lock(&self.state);
        st.wanted = None;
        if let Some(run) = st.running.as_ref().filter(|r| r.speculative) {
            run.stop.cancel();
        }
    }

    /// Whether the worker is making the map for the photo ("Finding…").
    pub(crate) fn working(&self) -> bool {
        lock(&self.state).finding
    }

    /// Whether a worker thread is alive for the photo, even one that only waits out the debounce.
    pub(crate) fn worker_alive(&self) -> bool {
        lock(&self.state).worker
    }

    /// How many times the network finished for the photo.
    pub(crate) fn networks(&self) -> u64 {
        lock(&self.state).networks
    }

    /// How many workers were started for the photo.
    pub(crate) fn workers(&self) -> u64 {
        lock(&self.state).workers
    }

    /// The map the photo has, whatever picture it was made from.
    pub(crate) fn latest(&self) -> Option<Arc<DepthEntry>> {
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

    /// The key of the map the photo has, if any.
    pub(crate) fn ready_key(&self) -> Option<String> {
        lock(&self.state).ready.as_ref().map(|e| e.key.clone())
    }
}

/// What a request remembers of its Depth rows.
#[derive(Default)]
pub(crate) struct JobDepth {
    key: Option<DepthKey>,
    map: Option<Arc<DepthMap>>,
}

// ---------------------------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------------------------

impl Renderer {
    /// The Depth model this renderer has, when its weights are there and have not been found damaged.
    #[must_use]
    pub fn depth_model(&self) -> Option<Arc<DepthModel>> {
        self.options().depth_model.clone().filter(|m| m.available())
    }

    /// Whether this renderer can make Depth masks.
    #[must_use]
    pub fn depth_available(&self) -> bool {
        self.depth_model().is_some()
    }

    /// Who hears when the depth worker starts or ends its work for a photo ([`DepthPhase`]). Replaces the one set before.
    pub fn set_depth_listener(&self, f: DepthListener) {
        *lock(&self.inner.depth_listener) = Some(f);
    }

    fn depth_notify(&self, id: PhotoId, phase: DepthPhase) {
        let listener = lock(&self.inner.depth_listener).clone();
        if let Some(f) = listener {
            f(id, phase);
        }
    }

    /// Whether the worker is making the depth map for the photo ("Finding…").
    #[must_use]
    pub fn depth_working(&self, id: PhotoId) -> bool {
        self.existing_graph(id).is_some_and(|g| g.depth.working())
    }

    /// Whether a background worker thread is alive for the photo, including one that only waits out the debounce.
    #[must_use]
    pub fn depth_worker_alive(&self, id: PhotoId) -> bool {
        self.existing_graph(id)
            .is_some_and(|g| g.depth.worker_alive())
    }

    /// How many times the network finished for the photo since its graph was made.
    #[must_use]
    pub fn depth_networks(&self, id: PhotoId) -> u64 {
        self.existing_graph(id).map_or(0, |g| g.depth.networks())
    }

    /// How many background workers were started for the photo since its graph was made.
    #[must_use]
    pub fn depth_workers(&self, id: PhotoId) -> u64 {
        self.existing_graph(id).map_or(0, |g| g.depth.workers())
    }

    /// Whether the photo has a map for the picture of these settings (a click would not wait).
    #[must_use]
    pub fn depth_ready(&self, src: &PhotoSource) -> bool {
        let Ok(key) = self.depth_key(src) else {
            return false;
        };
        self.existing_graph(src.id)
            .is_some_and(|g| g.depth.ready_key().as_deref() == Some(key.text.as_str()))
    }

    /// The sentence of the failure the photo keeps for the picture of these settings ("The Depth mask can't be made: …"),
    /// when its map could not be made.
    #[must_use]
    pub fn depth_failure(&self, src: &PhotoSource) -> Option<String> {
        let key = self.depth_key(src).ok()?;
        self.existing_graph(src.id)?.depth.failure(&key.text)
    }

    /// What the photo's depth map is made from: the picture (the one Object reads: the photo, the raw group's numbers and
    /// the Input's colour, and nothing a node, the look, the crop, the turn or the lens character says), the weights and
    /// the rules. Equal keys are equal maps.
    ///
    /// # Errors
    ///
    /// [`RenderError::Color`] for estimates that cannot make a picture, and [`RenderError::Mask`] with
    /// [`NO_DEPTH_MODEL_SENTENCE`] when the renderer has no model.
    pub fn depth_key(&self, src: &PhotoSource) -> Result<DepthKey, RenderError> {
        let model = self
            .depth_model()
            .ok_or_else(|| RenderError::Mask(NO_DEPTH_MODEL_SENTENCE.to_owned()))?;
        let picture_key = self.object_key(src)?;
        Ok(DepthKey {
            text: format!(
                "{picture_key}|depth|{}|f{DEPTH_FORMAT_VERSION}|r{DEPTH_RULES_VERSION}",
                model.weights_id()
            ),
            picture: fnv(&picture_key),
        })
    }

    /// Makes the photo's map for `key`: the picture on the CPU reference path (the one Object reads, kept), then the network
    /// on the model's threads. `cancel` stops the picture and `also` (asked between the network's blocks) the network.
    fn make_depth_entry(
        &self,
        model: &DepthModel,
        src: &PhotoSource,
        key: &DepthKey,
        cancel: &CancelToken,
        also: &(dyn Fn() -> bool + Sync),
    ) -> Result<DepthEntry, RenderError> {
        let started = Instant::now();
        let picture = self.object_picture(src, key.picture, cancel)?;
        let picture_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stopped = || cancel.is_cancelled() || also();
        let (width, height, d) =
            model.map_of(&picture.rgb, picture.width, picture.height, &stopped)?;
        if cancel.is_cancelled() {
            return Err(RenderError::Cancelled);
        }
        let map = DepthMap::new(fnv(&key.text), width, height, d)
            .map_err(|e| RenderError::Mask(format!("The Depth mask can't be made: {e}.")))?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(
            ms,
            picture_ms,
            "the depth map (the picture and the network)"
        );
        Ok(DepthEntry {
            key: key.text.clone(),
            map: Arc::new(map),
            ms,
        })
    }

    /// The photo's map for `key`, made when it has none: waits for the network when it is running for this key, stops a run
    /// that was started in the background for another one, and runs the network itself when nobody does. A map that cannot
    /// be made is kept as the failure of its key.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` is cancelled (or a run of this call was stopped because the person left the
    /// photo and the call is a background one), and the errors of the picture and the model.
    fn ensure_depth_entry(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &DepthModel,
        src: &PhotoSource,
        key: &DepthKey,
        speculative: bool,
        cancel: &CancelToken,
    ) -> Result<Arc<DepthEntry>, RenderError> {
        let slot = &graph.depth;
        loop {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            let mut st = lock(&slot.state);
            if let Some(ready) = st.ready.as_ref().filter(|e| e.key == key.text) {
                return Ok(Arc::clone(ready));
            }
            if let Some(run) = &st.running {
                // A run for another picture that nobody asked for in words is stopped; one a click is waiting for is not.
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
            let result = self.make_depth_entry(model, src, key, &stop, &|| cancel.is_cancelled());
            let mut st = lock(&slot.state);
            st.running = None;
            match result {
                Ok(entry) => {
                    let entry = Arc::new(entry);
                    st.ready = Some(Arc::clone(&entry));
                    st.failed = None;
                    st.networks += 1;
                    slot.cv.notify_all();
                    return Ok(entry);
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
                    st.failed = Some((key.text.clone(), e.to_string()));
                    slot.cv.notify_all();
                    return Err(e);
                }
            }
        }
    }

    /// Asks for the depth map of the photo's picture in the background (`docs/23` section 4.8: when a Depth row appears
    /// on the photo). Returns at once. Nothing is done when the model is missing, the photo has the map, the network is
    /// working for it, it failed for this picture, or a worker already wants it; a worker waits until nobody has changed
    /// the wish for a moment (150 ms) so a stream of requests makes one map.
    pub fn depth_prepare(&self, src: &PhotoSource) {
        let Some(model) = self.depth_model() else {
            return;
        };
        let Ok(key) = self.depth_key(src) else {
            return;
        };
        let graph = self.graph(src.id);
        self.depth_prepare_keyed(&graph, &model, src, key);
    }

    fn depth_prepare_keyed(
        &self,
        graph: &Arc<PhotoGraph>,
        model: &Arc<DepthModel>,
        src: &PhotoSource,
        key: DepthKey,
    ) {
        {
            let mut st = lock(&graph.depth.state);
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
            .name("yellowmoon-depth-prepare".to_owned())
            .spawn(move || renderer.depth_worker(&worker_graph, &worker_model));
        if let Err(e) = spawned {
            tracing::error!(error = %e, "the Depth worker could not start");
            let mut st = lock(&graph.depth.state);
            st.worker = false;
            st.wanted = None;
        }
    }

    /// The background worker of a photo: makes what is wanted, one picture at a time, until nothing is. A wish is taken once
    /// it has been left alone for [`PREPARE_DEBOUNCE`], so a stream of them (a drag of the Input's exposure) makes one map,
    /// of the last picture. The listener hears [`DepthPhase::Started`] when the first run begins and
    /// [`DepthPhase::Finished`] when the worker ends; a worker whose wishes were all dropped before a run began (the person
    /// left the photo) ends without a word.
    fn depth_worker(&self, graph: &Arc<PhotoGraph>, model: &Arc<DepthModel>) {
        let id = graph.id();
        let mut announced = false;
        let mut failed: Option<String> = None;
        loop {
            let wanted = {
                let mut st = lock(&graph.depth.state);
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
            lock(&graph.depth.state).finding = true;
            if !announced {
                announced = true;
                self.depth_notify(id, DepthPhase::Started);
            }
            match self.ensure_depth_entry(
                graph,
                model,
                &wanted.src,
                &wanted.key,
                true,
                &CancelToken::new(),
            ) {
                Ok(e) => {
                    failed = None;
                    tracing::debug!(ms = e.ms, "the depth map is ready");
                }
                Err(RenderError::Cancelled) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "the depth map could not be made");
                    failed = Some(e.to_string());
                }
            }
        }
        if announced {
            self.depth_notify(id, DepthPhase::Finished { failed });
        }
    }

    /// The photo is no longer the one being looked at: the worker stops working for it if it started on its own.
    pub(crate) fn depth_leave(&self, id: PhotoId) {
        if let Some(g) = self.existing_graph(id) {
            g.depth.abandon();
        }
    }

    /// The photo's map for the picture of these settings, waiting for the worker when it runs and making it when nobody
    /// does. `None` when it cannot be made (the failure is kept, [`Renderer::depth_failure`]).
    fn depth_entry_waiting(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Option<Arc<DepthEntry>>, RenderError> {
        let model = self
            .depth_model()
            .ok_or_else(|| RenderError::Mask(NO_DEPTH_MODEL_SENTENCE.to_owned()))?;
        let key = self.depth_key(src)?;
        let graph = self.graph(src.id);
        if graph.depth.failure(&key.text).is_some() {
            return Ok(None);
        }
        match self.ensure_depth_entry(&graph, &model, src, &key, false, cancel) {
            Ok(entry) => Ok(Some(entry)),
            Err(RenderError::Cancelled) => Err(RenderError::Cancelled),
            Err(e) => {
                tracing::warn!(error = %e, "the depth map could not be made for a click");
                Ok(None)
            }
        }
    }

    /// The photo's depth map for the picture of these settings, made now when the photo has none (waiting for the worker
    /// when it runs): for checks and diagnostics, like [`Renderer::blur_map`].
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when there is no model or the map cannot be made; [`RenderError::Cancelled`]; the errors of the
    /// picture.
    pub fn depth_map(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Arc<DepthMap>, RenderError> {
        match self.depth_entry_waiting(src, cancel)? {
            Some(entry) => Ok(Arc::clone(&entry.map)),
            None => Err(RenderError::Mask(
                self.depth_failure(src)
                    .unwrap_or_else(|| "The Depth mask can't be made.".to_owned()),
            )),
        }
    }

    /// `sample_depth` (`docs/23` section 9, Pick depth): the band around the depth at the point `(x, y)` of the photo frame
    /// (shares of its width and height), `(near, far)` rounded to two decimals (`docs/algorithms/06` section 5.8, step 4), or
    /// `None` when no map can be made. It waits for the worker when it runs and reads the cached map; it changes nothing:
    /// the caller applies the answer as one row patch.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when there is no model or the point is not a number; [`RenderError::Cancelled`]; the errors of
    /// the picture.
    pub fn sample_depth(
        &self,
        src: &PhotoSource,
        x: f64,
        y: f64,
        cancel: &CancelToken,
    ) -> Result<Option<(f64, f64)>, RenderError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(RenderError::Mask(BAD_POINT.to_owned()));
        }
        Ok(self
            .depth_entry_waiting(src, cancel)?
            .map(|entry| pick_depth(&entry.map, x, y)))
    }

    /// `focus_plane` (`docs/23` section 9, Focus plane): the band around the plane the camera focused on, from the maker
    /// note's focus point and the median depth in its box (`docs/algorithms/06` section 5.8, step 5). A photo with no focus
    /// point is answered at once, without waiting for the map. It changes nothing: the caller applies the answer as one row
    /// patch.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when there is no model; [`RenderError::Cancelled`]; the errors of the photo's geometry and of
    /// the picture.
    pub fn focus_plane(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<FocusPlane, RenderError> {
        if self.depth_model().is_none() {
            return Err(RenderError::Mask(NO_DEPTH_MODEL_SENTENCE.to_owned()));
        }
        let focus = match src.raw.meta.focus {
            // Sony writes 0 0 for a panorama: there is no point.
            Some(f) if f.point != [0, 0] => f,
            _ => return Ok(FocusPlane::None(FocusReason::NoFocusPoint)),
        };
        let exact_centre = 2 * u32::from(focus.point[0]) == u32::from(focus.frame[0])
            && 2 * u32::from(focus.point[1]) == u32::from(focus.frame[1]);
        // The frame the point is in: the sensor's size and R9's crop, the frame the lens works on. It is the photo
        // frame the map covers, before the user's crop and turn.
        let frame = {
            let reference = self.object_reference()?;
            let stripped = object_source(src);
            let mut job = Job::new(reference, &stripped, cancel)?;
            let frame = job.focus_frame_of();
            drop(job);
            reference.forget(src.id);
            frame?
        };
        let Some(frame) = frame else {
            return Ok(FocusPlane::None(FocusReason::NoFocusPoint));
        };
        let Some(entry) = self.depth_entry_waiting(src, cancel)? else {
            return Ok(FocusPlane::None(FocusReason::MapFailed));
        };
        let (near, far) = focus_band(&entry.map, &frame, src.exif_orientation());
        Ok(FocusPlane::Band {
            near,
            far,
            exact_centre,
        })
    }
}

/// The band `(near, far)` around the plane the camera focused on: the median depth of the map inside the AF box
/// (`docs/algorithms/06` section 5.8, step 5). `frame` is the maker note's point and box in the auto-cropped frame in the
/// sensor's orientation; both are carried to the photo as the camera oriented it (`exif_orientation`, 1 to 8), because the
/// map is in that frame. Without a box the box is a square of [`FOCUS_BOX_DEFAULT`] of the photo's long edge.
pub(crate) fn focus_band(map: &DepthMap, frame: &FocusFrame, exif_orientation: u16) -> (f64, f64) {
    let [cx, cy] = sensor_to_oriented(exif_orientation, [frame.point.0, frame.point.1]);
    let (cells_w, cells_h) = (map.width as f64, map.height as f64);
    let (bw, bh) = match frame.box_size {
        // The axes of the box turn with the photo.
        Some((w, h)) if (5..=8).contains(&exif_orientation) => (h, w),
        Some((w, h)) => (w, h),
        None => {
            let side = FOCUS_BOX_DEFAULT * cells_w.max(cells_h);
            (side / cells_w, side / cells_h)
        }
    };
    pick_band(map.focus_z(cx, cy, bw, bh))
}

// ---------------------------------------------------------------------------------------------
// A request
// ---------------------------------------------------------------------------------------------

impl Job<'_> {
    /// The photo's depth map for the plan that ends at `stop`, when that plan has a Depth row with the eye on. `None` for
    /// a plan with none, when this machine has no model, and (for a viewer) when the photo has no map yet: the worker is
    /// started in the background and the rows select nothing until the listener tells the page to draw again. A viewer
    /// keeps the latest map of another picture meanwhile. An export makes the map itself when the photo has none of the right
    /// key (and does not keep it).
    ///
    /// # Errors
    ///
    /// For an export, [`RenderError::Mask`] with "The Depth mask can't be made: …"; [`RenderError::Cancelled`]; the errors of
    /// the picture.
    pub(crate) fn depth_for(
        &mut self,
        settings: &DevelopSettings,
        stop: PlanStop,
    ) -> Result<Option<Arc<DepthMap>>, RenderError> {
        if !uses_depth(settings, stop) {
            return Ok(None);
        }
        let Some(model) = self.opts.depth_model.clone().filter(|m| m.available()) else {
            return Ok(None);
        };
        if let Some(map) = &self.depth.map {
            return Ok(Some(Arc::clone(map)));
        }
        let key = match &self.depth.key {
            Some(k) => k.clone(),
            None => {
                let k = self.r.depth_key(self.src)?;
                self.depth.key = Some(k.clone());
                k
            }
        };
        let ready = self.g.depth.latest();
        if let Some(e) = ready.as_ref().filter(|e| e.key == key.text) {
            self.depth.map = Some(Arc::clone(&e.map));
            return Ok(Some(Arc::clone(&e.map)));
        }
        if self.restartable {
            self.check()?;
            let cancel = self.cancel;
            let made = self
                .r
                .make_depth_entry(&model, self.src, &key, cancel, &|| false)?;
            self.depth.map = Some(Arc::clone(&made.map));
            return Ok(Some(made.map));
        }
        // A map that failed for this picture is not asked for again; the row selects nothing.
        if self.g.depth.failure(&key.text).is_none() {
            self.r.depth_prepare_keyed(&self.g, &model, self.src, key);
        }
        // Meanwhile the latest map, from another picture: it stays put while the picture moves.
        Ok(ready.map(|e| Arc::clone(&e.map)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A map of 20 x 10 cells whose nearest cell is `(cx, cy)`, the rest at one depth.
    fn map_with_near_cell(cx: usize, cy: usize) -> DepthMap {
        let mut d = vec![0.0f32; 200];
        d[cy * 20 + cx] = 1.0;
        DepthMap::new(1, 20, 10, d).expect("a map")
    }

    /// The scene point at `(0.3, 0.7)` of the sensor is where the oriented photo has it (`sensor_to_oriented`): in the eight
    /// orientations a box on it lands on the block of near cells the photo has there (a median of near cells), and a box
    /// anywhere else lands on the far ones.
    #[test]
    fn the_box_lands_on_the_same_scene_point_in_the_eight_orientations() {
        for orientation in 1..=8u16 {
            let [ox, oy] = sensor_to_oriented(orientation, [0.3, 0.7]);
            let swapped = (5..=8).contains(&orientation);
            let (w, h) = if swapped {
                (10usize, 20usize)
            } else {
                (20, 10)
            };
            let (cx, cy) = (
                ((ox * w as f64) as usize).min(w - 1),
                ((oy * h as f64) as usize).min(h - 1),
            );
            // A block of near cells around the scene point, five cells across.
            let mut d = vec![0.0f32; w * h];
            for y in cy.saturating_sub(2)..(cy + 3).min(h) {
                for x in cx.saturating_sub(2)..(cx + 3).min(w) {
                    d[y * w + x] = 1.0;
                }
            }
            let map = DepthMap::new(1, w, h, d).expect("a map");
            let on = FocusFrame {
                point: (0.3, 0.7),
                box_size: Some((0.15, 0.15)),
            };
            assert_eq!(
                focus_band(&map, &on, orientation),
                (0.0, 0.2),
                "orientation {orientation}"
            );
            // The same point read as if the photo had not been turned lands elsewhere (but for the orientation 1, and the
            // turns that leave this point where it is): the far cells.
            let off = FocusFrame {
                point: (0.8, 0.2),
                box_size: Some((0.15, 0.15)),
            };
            let band = focus_band(&map, &off, orientation);
            assert!(band.0 >= 0.4, "orientation {orientation}: {band:?}");
        }
    }

    #[test]
    fn without_a_box_the_square_is_four_percent_of_the_long_edge_and_at_least_three_cells() {
        // 20 x 10 cells: 4 % of 20 is under one cell, so the box is the minimum, 3 cells: the nearest cell with its eight
        // neighbours around it, and the median of nine cells with one near is the others'.
        let map = map_with_near_cell(10, 5);
        let frame = FocusFrame {
            point: (0.525, 0.55),
            box_size: None,
        };
        let band = focus_band(&map, &frame, 1);
        assert!(band.0 >= 0.4, "{band:?}");
        // A point that is not inside the frame's cells still answers.
        let edge = FocusFrame {
            point: (1.0, 1.0),
            box_size: None,
        };
        let band = focus_band(&map, &edge, 1);
        assert!((0.0..=1.0).contains(&band.0) && (0.0..=1.0).contains(&band.1));
    }
}

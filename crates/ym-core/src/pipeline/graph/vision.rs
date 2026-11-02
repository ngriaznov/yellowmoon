//! Subject and People in the render graph (`docs/23-nodes.md` sections 3.10, 4.5 and 5.6,
//! `docs/algorithms/06-masks.md` section 5.3).
//!
//! A Subject or People row's raster `S0` is the **union of the instances** that one Vision request finds in the Input's
//! picture: the same picture the Object mask reads ([`object_source`](crate::pipeline::object::object_source) rendered
//! by [`Renderer::object_picture`]: the photo frame, the Input's colour, the Neutral look, 8 bits, sRGB, 1024 px on the
//! long edge, no output sharpening). There is no second picture path.
//!
//! * **The masks live in the photo's graph** ([`VisionSlot`]), in memory, keyed by the picture's key
//!   ([`Renderer::object_key`]: the raw group's numbers and the Input's colour, never a node's grade or the look) and
//!   the request. Nothing goes to disk. Grading a node, changing the look or moving a slider of another node never makes a
//!   mask again; the Input's exposure does, as it moves the picture.
//! * **A viewer never waits.** A frame draws the masks the photo has. For a mask it has not, it asks for one in the
//!   background ([`Renderer::vision_prepare`]: one worker per photo that waits a moment, so a drag of the Input's
//!   exposure makes one mask, not a hundred) and the row selects nothing until the listener ([`VisionPhase`]) tells the
//!   page to draw again. While the picture changes it keeps the latest mask of the request it has, as Object does, so the
//!   mask stays put during a drag. The app says "Finding…" under the row meanwhile (`viewport_busy.finding`).
//! * **An export makes its own mask** from the same picture when the photo has none for it, so its bytes depend on
//!   nothing but the file and the settings (and this Mac's Vision), and `ym` and the app agree.
//! * **Nothing found** is a result too: `S0` is 0 (the row has no plane) and the page says "Nothing found". A request
//!   that fails is kept as a failure, so a redraw does not ask again; an export asks again itself and fails with the
//!   `mask` error if it fails too.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ym_color::mask_raster::SourcePlane;
use ym_color::plan::Stop as PlanStop;
use ym_gpu::queue::CancelToken;

use super::object::fnv;
use super::{Job, PhotoGraph, Renderer, lock};
use crate::apple_auto::AdvisorPicture;
use crate::pipeline::RenderError;
use crate::pipeline::color::{PhotoSource, vision_kinds, vision_kinds_to};
use crate::pipeline::vision::VisionPhase;
use crate::session::PhotoId;
use crate::settings::{MaskKind, MaskSupport, MaskUnavailable};
use crate::vision::{
    VisionError, VisionKind, VisionMask, VisionMasks, VisionSupport, plane_codes, plane_key,
};

/// How long the background worker waits before it makes a mask, so that a drag of the Input's exposure (which asks for a
/// new picture on every tick) makes one mask, not a hundred (the same wait as the Object encoder's).
pub(crate) const PREPARE_DEBOUNCE: Duration = super::object::PREPARE_DEBOUNCE;

/// How many results a photo keeps (two requests on a few pictures; a plane is 1.4 MB at 1024 x 683).
const ENTRIES_KEPT: usize = 8;

/// Who hears when the background worker starts or ends its work for a photo ([`Renderer::set_vision_listener`]).
pub(crate) type VisionListener = Arc<dyn Fn(PhotoId, VisionPhase) + Send + Sync>;

/// What a request made of a picture.
#[derive(Debug, Clone)]
pub(crate) enum VisionResult {
    /// The union of the instances, as u16 codes.
    Found(Arc<SourcePlane>),
    /// Vision found no instance: `S0` is 0.
    NothingFound,
    /// Vision failed (the reason is in the log); not asked again by a viewer.
    Failed,
}

/// A result with what it is for.
#[derive(Debug)]
pub(crate) struct VisionEntry {
    /// The hash of the picture's key ([`Renderer::object_key`]).
    picture: u64,
    kind: VisionKind,
    result: VisionResult,
}

/// A wish of the background worker: these requests on this photo's picture.
struct Wanted {
    hash: u64,
    src: PhotoSource,
    kinds: Vec<VisionKind>,
    /// When the wish was made: the worker starts once nobody has changed it for [`PREPARE_DEBOUNCE`].
    at: Instant,
}

/// The requests in flight, so that a newer wish for another picture can stop a run nobody is waiting for.
struct Running {
    hash: u64,
    kinds: Vec<VisionKind>,
    stop: CancelToken,
}

#[derive(Default)]
struct SlotState {
    /// The results made lately, newest last.
    entries: VecDeque<Arc<VisionEntry>>,
    /// What the background worker will make next.
    wanted: Option<Wanted>,
    /// The worker is making masks now.
    running: Option<Running>,
    /// A background worker is alive for the photo.
    worker: bool,
    /// How many requests reached the provider for the photo, workers and exports together (for the tests and the log).
    makes: u64,
}

/// What a photo's graph keeps for its Subject and People rows.
#[derive(Default)]
pub(crate) struct VisionSlot {
    state: std::sync::Mutex<SlotState>,
}

impl VisionSlot {
    /// Bytes the planes hold.
    pub(crate) fn bytes(&self) -> usize {
        lock(&self.state)
            .entries
            .iter()
            .map(|e| match &e.result {
                VisionResult::Found(p) => 2 * p.data.len(),
                _ => 0,
            })
            .sum()
    }

    /// The person left the photo: nothing is wanted for it, and a run in the background stops (between its picture
    /// and its requests; a request Vision is running cannot be stopped, and its result is kept).
    pub(crate) fn abandon(&self) {
        let mut st = lock(&self.state);
        st.wanted = None;
        if let Some(run) = &st.running {
            run.stop.cancel();
        }
    }

    /// Whether the worker is making masks for the photo, or will be in a moment ("Finding…").
    pub(crate) fn working(&self) -> bool {
        let st = lock(&self.state);
        st.worker || st.running.is_some()
    }

    /// How many requests reached the provider for the photo.
    pub(crate) fn makes(&self) -> u64 {
        lock(&self.state).makes
    }

    /// The result for this picture and request, if the photo has one.
    fn entry(&self, picture: u64, kind: VisionKind) -> Option<Arc<VisionEntry>> {
        lock(&self.state)
            .entries
            .iter()
            .rev()
            .find(|e| e.picture == picture && e.kind == kind)
            .cloned()
    }

    /// The newest result of a request, for any picture.
    fn latest(&self, kind: VisionKind) -> Option<Arc<VisionEntry>> {
        lock(&self.state)
            .entries
            .iter()
            .rev()
            .find(|e| e.kind == kind)
            .cloned()
    }

    /// Keeps a result, replacing the one for the same picture and request.
    fn keep(&self, entry: VisionEntry) {
        let mut st = lock(&self.state);
        st.entries
            .retain(|e| !(e.picture == entry.picture && e.kind == entry.kind));
        st.entries.push_back(Arc::new(entry));
        while st.entries.len() > ENTRIES_KEPT {
            st.entries.pop_front();
        }
    }

    /// The requests whose newest result found nothing.
    pub(crate) fn nothing_found(&self) -> Vec<MaskKind> {
        VisionKind::ALL
            .into_iter()
            .filter(|&k| {
                self.latest(k)
                    .is_some_and(|e| matches!(e.result, VisionResult::NothingFound))
            })
            .map(VisionKind::mask_kind)
            .collect()
    }
}

/// What a request remembers of its Subject and People rows.
#[derive(Default)]
pub(crate) struct JobVision {
    /// The picture's key and its hash, once asked for.
    key: Option<(String, u64)>,
    /// The results this request has for the picture it asked about: a plane, or none for nothing found.
    known: std::collections::BTreeMap<VisionKind, Option<Arc<SourcePlane>>>,
}

/// The sentence for a request that failed in an export: what went wrong and what to do.
fn failed_sentence(kind: VisionKind, why: &VisionError) -> String {
    match why {
        VisionError::Unavailable(..) => format!(
            "This computer cannot make {} masks. Use a Mac that supports {} masks, or remove the mask.",
            kind.label(),
            kind.label()
        ),
        _ => format!(
            "The {} mask can't be made: {}. Try again, or remove the mask.",
            kind.label(),
            why.to_string().trim_end_matches('.')
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------------------------

impl Renderer {
    /// The provider of Subject and People masks this renderer was given and its probe answer.
    #[must_use]
    pub fn vision_masks(&self) -> Option<Arc<VisionMasks>> {
        self.options().vision.clone()
    }

    /// What this computer can do for Subject and People (`docs/23` section 3.10): the probe, run the first time this is
    /// asked and kept for the run. A renderer made without a provider says what a build with no Vision code says
    /// ([`VisionSupport::none_here`]).
    #[must_use]
    pub fn vision_support(&self) -> VisionSupport {
        self.vision_masks()
            .map_or_else(VisionSupport::none_here, |v| v.support())
    }

    /// What this computer can make now: the SAM weights, and what the Vision probe said.
    #[must_use]
    pub fn mask_support(&self) -> MaskSupport {
        MaskSupport::new(self.object_available(), self.vision_support())
    }

    /// Who hears when the background worker starts or ends its work for a photo ([`VisionPhase`]). Replaces the one set
    /// before.
    pub fn set_vision_listener(&self, f: VisionListener) {
        *lock(&self.inner.vision_listener) = Some(f);
    }

    fn vision_notify(&self, id: PhotoId, phase: VisionPhase) {
        let listener = lock(&self.inner.vision_listener).clone();
        if let Some(f) = listener {
            f(id, phase);
        }
    }

    /// Whether the worker is making Subject or People masks for the photo, or will be in a moment ("Finding…").
    #[must_use]
    pub fn vision_working(&self, id: PhotoId) -> bool {
        self.existing_graph(id).is_some_and(|g| g.vision.working())
    }

    /// How many requests reached the provider for the photo since its graph was made.
    #[must_use]
    pub fn vision_makes(&self, id: PhotoId) -> u64 {
        self.existing_graph(id).map_or(0, |g| g.vision.makes())
    }

    /// The requests whose newest result for the photo found nothing (the rows say "Nothing found").
    #[must_use]
    pub fn vision_nothing_found(&self, id: PhotoId) -> Vec<MaskKind> {
        self.existing_graph(id)
            .map_or_else(Vec::new, |g| g.vision.nothing_found())
    }

    /// Asks for the Subject and People masks of the photo's picture in the background (`docs/23` section 4.5: "when the
    /// row appears"). Returns at once, probe included: the worker asks the computer. Nothing is done when the renderer has
    /// no provider, the photo has no such row, or the masks are made already or being made.
    pub fn vision_prepare(&self, src: &PhotoSource) {
        let Some(masks) = self.vision_masks() else {
            return;
        };
        let mut kinds = vision_kinds(&src.settings);
        // Once the probe is known, a request this computer cannot run is not asked for.
        if let Some(known) = masks.support_if_known() {
            kinds.retain(|&k| known.of(k).is_supported());
        }
        if kinds.is_empty() {
            return;
        }
        let Ok(key) = self.object_key(src) else {
            return;
        };
        let graph = self.graph(src.id);
        self.vision_prepare_keyed(&graph, &masks, src, &key, &kinds);
    }

    fn vision_prepare_keyed(
        &self,
        graph: &Arc<PhotoGraph>,
        masks: &Arc<VisionMasks>,
        src: &PhotoSource,
        key: &str,
        kinds: &[VisionKind],
    ) {
        let hash = fnv(key);
        {
            let mut st = lock(&graph.vision.state);
            let mut missing: Vec<VisionKind> = kinds
                .iter()
                .copied()
                .filter(|&k| !st.entries.iter().any(|e| e.picture == hash && e.kind == k))
                .collect();
            if missing.is_empty() {
                return;
            }
            if st
                .running
                .as_ref()
                .is_some_and(|r| r.hash == hash && missing.iter().all(|k| r.kinds.contains(k)))
            {
                return;
            }
            if let Some(w) = st.wanted.as_ref().filter(|w| w.hash == hash) {
                for k in &w.kinds {
                    if !missing.contains(k) {
                        missing.push(*k);
                    }
                }
            }
            missing.sort_unstable();
            // A run made in the background for another picture is stopped: the newer wish replaces it.
            if let Some(run) = st.running.as_ref().filter(|r| r.hash != hash) {
                run.stop.cancel();
            }
            st.wanted = Some(Wanted {
                hash,
                src: src.clone(),
                kinds: missing,
                at: Instant::now(),
            });
            if st.worker {
                return;
            }
            st.worker = true;
        }
        let renderer = self.clone();
        let (worker_graph, worker_masks) = (Arc::clone(graph), Arc::clone(masks));
        let spawned = std::thread::Builder::new()
            .name("yellowmoon-vision".to_owned())
            .spawn(move || renderer.vision_worker(&worker_graph, &worker_masks));
        if let Err(e) = spawned {
            tracing::error!(error = %e, "the Vision worker could not start");
            let mut st = lock(&graph.vision.state);
            st.worker = false;
            st.wanted = None;
        }
    }

    /// The background worker of a photo: asks the computer what it supports, then makes what is wanted, one picture at a
    /// time, until nothing is. A wish is taken once it has been left alone for [`PREPARE_DEBOUNCE`], so a stream of them
    /// makes one mask, of the last picture.
    fn vision_worker(&self, graph: &Arc<PhotoGraph>, masks: &Arc<VisionMasks>) {
        let id = graph.id();
        self.vision_notify(id, VisionPhase::Started);
        loop {
            let wanted = {
                let mut st = lock(&graph.vision.state);
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
            // The probe runs here, off the thread that asked: once per run.
            let support = masks.support();
            let kinds: Vec<VisionKind> = wanted
                .kinds
                .iter()
                .copied()
                .filter(|&k| support.of(k).is_supported())
                .filter(|&k| graph.vision.entry(wanted.hash, k).is_none())
                .collect();
            if kinds.is_empty() {
                continue;
            }
            let stop = CancelToken::new();
            lock(&graph.vision.state).running = Some(Running {
                hash: wanted.hash,
                kinds: kinds.clone(),
                stop: stop.clone(),
            });
            let made = self.make_vision(masks, &wanted.src, &kinds, wanted.hash, &stop);
            {
                let mut st = lock(&graph.vision.state);
                st.running = None;
                st.makes += made.as_ref().map_or(0, |m| m.len() as u64);
            }
            match made {
                Ok(results) => {
                    for (kind, result) in results {
                        let result = result.unwrap_or_else(|e| {
                            tracing::warn!(error = %e, kind = kind.label(), "the Vision mask could not be made");
                            VisionResult::Failed
                        });
                        graph.vision.keep(VisionEntry {
                            picture: wanted.hash,
                            kind,
                            result,
                        });
                    }
                }
                Err(RenderError::Cancelled) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "the picture for the Vision masks could not be made")
                }
            }
        }
        self.vision_notify(
            id,
            VisionPhase::Finished {
                nothing_found: graph.vision.nothing_found(),
            },
        );
    }

    /// The photo is no longer the one being looked at: the worker stops working for it.
    pub(crate) fn vision_leave(&self, id: PhotoId) {
        if let Some(g) = self.existing_graph(id) {
            g.vision.abandon();
        }
    }

    /// Makes the masks of these requests on the photo's picture: the picture once, on the CPU reference path
    /// ([`Renderer::object_picture`]), then each request. A request's own failure is in its slot; the picture's is the
    /// error. `hash` names the picture ([`Renderer::object_key`]).
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` is cancelled, and the errors of the picture.
    fn make_vision(
        &self,
        masks: &VisionMasks,
        src: &PhotoSource,
        kinds: &[VisionKind],
        hash: u64,
        cancel: &CancelToken,
    ) -> Result<Vec<(VisionKind, Result<VisionResult, VisionError>)>, RenderError> {
        let started = Instant::now();
        let picture: AdvisorPicture = self.object_picture(src, cancel)?;
        let picture_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut out = Vec::with_capacity(kinds.len());
        for &kind in kinds {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            let t = Instant::now();
            let result = masks.mask(kind, &picture).map(|made| match made {
                VisionMask::NothingFound => VisionResult::NothingFound,
                VisionMask::Plane(plane) => VisionResult::Found(Arc::new(SourcePlane {
                    key: plane_key(hash, kind),
                    width: plane.width,
                    height: plane.height,
                    data: plane_codes(&plane),
                })),
            });
            tracing::debug!(
                kind = kind.label(),
                picture_ms,
                ms = t.elapsed().as_secs_f64() * 1000.0,
                found = matches!(result, Ok(VisionResult::Found(_))),
                "a Vision mask"
            );
            out.push((kind, result));
        }
        Ok(out)
    }

    /// The plane of a Subject or People row for a photo, made now when the photo has none for its picture: for checks and
    /// diagnostics, like [`Renderer::object_plane`]. `None` when Vision finds nothing.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when this renderer has no provider or the computer cannot make the request, or Vision fails;
    /// [`RenderError::Cancelled`]; the errors of the picture.
    pub fn vision_plane(
        &self,
        src: &PhotoSource,
        kind: VisionKind,
        cancel: &CancelToken,
    ) -> Result<Option<Arc<SourcePlane>>, RenderError> {
        let Some(masks) = self.vision_masks() else {
            let why = VisionSupport::none_here()
                .of(kind)
                .reason()
                .unwrap_or(MaskUnavailable::NotSupported);
            return Err(RenderError::Mask(failed_sentence(
                kind,
                &VisionError::Unavailable(kind, why),
            )));
        };
        let key = self.object_key(src)?;
        let hash = fnv(&key);
        let graph = self.graph(src.id);
        if let Some(e) = graph.vision.entry(hash, kind)
            && !matches!(e.result, VisionResult::Failed)
        {
            return Ok(match &e.result {
                VisionResult::Found(p) => Some(Arc::clone(p)),
                _ => None,
            });
        }
        let mut made = self.make_vision(&masks, src, &[kind], hash, cancel)?;
        lock(&graph.vision.state).makes += 1;
        let (_, result) = made.remove(0);
        match result {
            Ok(result) => {
                let plane = match &result {
                    VisionResult::Found(p) => Some(Arc::clone(p)),
                    _ => None,
                };
                graph.vision.keep(VisionEntry {
                    picture: hash,
                    kind,
                    result,
                });
                Ok(plane)
            }
            Err(e) => Err(RenderError::Mask(failed_sentence(kind, &e))),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// A request
// ---------------------------------------------------------------------------------------------

impl Job<'_> {
    /// The planes of the Subject and People rows of the plan that ends at `stop`, one for each request the rows need and
    /// the computer can make (`None` for a request that found nothing, that is not made yet, or that this computer cannot
    /// make). A viewer never waits: it takes what the photo has, asks for the rest in the background, and meanwhile
    /// shows the latest mask of the request from another picture, if there is one. An export makes the masks itself when
    /// the photo has none for its picture.
    ///
    /// # Errors
    ///
    /// For an export, [`RenderError::Mask`] with the plain sentence for a request that fails; [`RenderError::Cancelled`];
    /// the errors of the picture.
    pub(crate) fn vision_planes_for(
        &mut self,
        stop: PlanStop,
    ) -> Result<Vec<(VisionKind, Arc<SourcePlane>)>, RenderError> {
        let wanted = vision_kinds_to(&self.src.settings, stop);
        if wanted.is_empty() {
            return Ok(Vec::new());
        }
        let Some(masks) = self.opts.vision.clone() else {
            return Ok(Vec::new());
        };
        // An export asks the computer and waits for the answer; a viewer reads what is known, and the worker asks.
        let support = if self.restartable {
            Some(masks.support())
        } else {
            masks.support_if_known()
        };
        let kinds: Vec<VisionKind> = wanted
            .into_iter()
            .filter(|&k| support.is_none_or(|s| s.of(k).is_supported()))
            .collect();
        if kinds.is_empty() {
            return Ok(Vec::new());
        }
        let (key, hash) = match &self.vision.key {
            Some(k) => k.clone(),
            None => {
                let key = self.r.object_key(self.src)?;
                let hash = fnv(&key);
                self.vision.key = Some((key.clone(), hash));
                (key, hash)
            }
        };
        let mut planes = Vec::new();
        let mut missing = Vec::new();
        for kind in kinds {
            if let Some(known) = self.vision.known.get(&kind) {
                planes.extend(known.clone().map(|p| (kind, p)));
                continue;
            }
            match self.g.vision.entry(hash, kind).map(|e| e.result.clone()) {
                Some(VisionResult::Found(p)) => {
                    self.vision.known.insert(kind, Some(Arc::clone(&p)));
                    planes.push((kind, p));
                }
                Some(VisionResult::NothingFound) => {
                    self.vision.known.insert(kind, None);
                }
                // A viewer does not ask again; an export does.
                Some(VisionResult::Failed) if !self.restartable => {}
                _ => missing.push(kind),
            }
        }
        if missing.is_empty() {
            return Ok(planes);
        }
        if self.restartable {
            self.check()?;
            let cancel = self.cancel;
            let made = self
                .r
                .make_vision(&masks, self.src, &missing, hash, cancel)?;
            for (kind, result) in made {
                match result {
                    Ok(VisionResult::Found(p)) => {
                        self.vision.known.insert(kind, Some(Arc::clone(&p)));
                        planes.push((kind, p));
                    }
                    Ok(_) => {
                        self.vision.known.insert(kind, None);
                    }
                    Err(e) => return Err(RenderError::Mask(failed_sentence(kind, &e))),
                }
            }
            lock(&self.g.vision.state).makes += missing.len() as u64;
        } else {
            self.r
                .vision_prepare_keyed(&self.g, &masks, self.src, &key, &missing);
            // Meanwhile the latest mask of the request, from another picture: it stays put while the picture moves.
            for kind in missing {
                if let Some(VisionResult::Found(p)) =
                    self.g.vision.latest(kind).map(|e| e.result.clone())
                {
                    planes.push((kind, p));
                }
            }
        }
        Ok(planes)
    }
}

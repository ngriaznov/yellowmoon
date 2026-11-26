//! The Reference Grade fit as the core runs it (`docs/34-reference-grade.md` sections 5 and 9, RG-06).
//!
//! The pure fit (the network, the measures, the objective, the search) is `ym_color::refgrade`; this module is the chain's side of it
//! and the commands around it:
//!
//! * [`ChainCandidates`] is the fit's **candidate renderer**: for numbers `x` it writes them into the node, puts the node into the
//!   photo's chain and runs the node, every per-pixel step after it, the Output's look, C12 and C13 on the 8192 sample pixels
//!   (their state at the node's input was made once, `Renderer::reference_fit_photo`), split in fixed chunks of 256 whatever the
//!   thread count. Every statistic is reduced afterwards, in pixel order, by the fit: **the same bits on one thread and on many**.
//! * [`Core::reference_choose_file`] and [`Core::reference_choose_photo`] choose a reference; the core **holds it per node for the
//!   session** (`docs/34` section 3.7: never saved, never written) and drops it with the node, the photo or the app.
//! * [`Core::reference_fit`] runs the fit for a node and writes its result as **one patch** of the node's fitted fields
//!   (`grade.curve`, `grade.saturation`, both hue curves and all seven element members): a fit starts again from the identity and
//!   leaves every other field alone. A result no better than the identity writes the identity ("already close"); a failure and a
//!   Cancel write nothing.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::prelude::*;
use ym_color::display::ColorChain;
use ym_color::mask::NodeMasks;
use ym_color::refgrade::measure::{LabConv, PixelScene, Px};
use ym_color::refgrade::network::Network;
use ym_color::refgrade::objective::Fixed;
use ym_color::refgrade::params::{ElementNumbers, Numbers, Params, to_numbers};
use ym_color::refgrade::search::{
    CandidateRenderer, FitError, FitProgress, FitResult, FitSetup, Refused, fit,
};
use ym_gpu::queue::CancelToken;

use crate::app::Core;
use crate::pipeline::color::reference_node_params;
use crate::pipeline::fit::FitPhoto;
use crate::pipeline::graph::ReferenceScene;
use crate::pipeline::{
    NO_SCENE_MODEL_SENTENCE, NO_SKIN_MODEL_SENTENCE, PhotoSource, RenderError, Renderer,
    SnapshotError,
};
use crate::reference::{ReferenceError, ReferenceNote, ReferencePicture, ReferenceSource};
use crate::session::{PhotoId, SessionError};
use crate::settings::{
    DevelopSettings, ElementValues, Elements, HueCurves, HuePoint, NodeId, NodeKind,
    PercentSetting, ReferenceGradeNode, SettingsPatch,
};

/// The fitted fields' identity curve (the node's default).
fn identity_curve() -> Vec<[f64; 2]> {
    crate::settings::Curves::default().luma
}

/// Writes a fit's numbers into the node's fitted fields (`docs/34` section 3.1): `grade.curve` (luma, red, green, blue),
/// `grade.saturation`, both hue curves and all seven element members. A curve whose numbers are all 0 is the node's default curve;
/// a hue curve that does nothing is the empty list. Nothing else of the node changes.
pub fn apply_numbers(node: &mut ReferenceGradeNode, n: &Numbers) {
    let curve = |c: &Option<Vec<[f64; 2]>>| c.clone().unwrap_or_else(identity_curve);
    node.grade.curve.luma = curve(&n.luma);
    node.grade.curve.red = curve(&n.rgb[0]);
    node.grade.curve.green = curve(&n.rgb[1]);
    node.grade.curve.blue = curve(&n.rgb[2]);
    node.grade.saturation = PercentSetting::manual_value(n.saturation_pct);
    let points = |values: &[f64; 6]| -> Vec<HuePoint> {
        ym_color::refgrade::params::HUE_KNOTS_DEG
            .iter()
            .zip(values)
            .map(|(&hue_deg, &value)| HuePoint { hue_deg, value })
            .collect()
    };
    node.hue_curves = HueCurves {
        hue_vs_hue: if n.hue_vs_hue_deg.iter().all(|&v| v == 0.0) {
            Vec::new()
        } else {
            points(&n.hue_vs_hue_deg)
        },
        hue_vs_sat: if n.hue_vs_sat.iter().all(|&v| v == 1.0) {
            Vec::new()
        } else {
            points(&n.hue_vs_sat)
        },
    };
    let values = |e: &ElementNumbers| ElementValues {
        exposure_ev: e.exposure_ev,
        hue_deg: e.hue_deg,
        chroma: e.chroma,
    };
    node.elements = Elements {
        sky: values(&n.elements[0]),
        vegetation: values(&n.elements[1]),
        water: values(&n.elements[2]),
        ground: values(&n.elements[3]),
        built: values(&n.elements[4]),
        skin: values(&n.elements[5]),
        subject: crate::settings::SubjectValues {
            exposure_ev: n.subject_ev,
        },
    };
}

/// The chain's candidate renderer (`docs/34` section 2.2).
pub struct ChainCandidates<'a> {
    photo: &'a FitPhoto,
    chain: ColorChain,
    base: ReferenceGradeNode,
    conv: LabConv,
}

impl<'a> ChainCandidates<'a> {
    /// A renderer for `photo` and the node as it stands (`node`: the person's trims, other grade fields and strength are part of the
    /// chain the fit matches through; the fit runs it at Strength 100, with no masks, not bypassed).
    #[must_use]
    pub fn new(photo: &'a FitPhoto, node: &ReferenceGradeNode) -> Self {
        let mut base = node.clone();
        base.bypass = false;
        base.strength = 100.0;
        Self {
            photo,
            chain: photo.chain.clone(),
            base,
            conv: LabConv::new(),
        }
    }

    /// The node the numbers `x` make: the fit's start, the person's other fields as they are.
    #[must_use]
    pub fn node_for(&self, x: &Params) -> ReferenceGradeNode {
        let mut node = self.base.clone();
        apply_numbers(&mut node, &to_numbers(x));
        node
    }
}

/// The chunk of sample pixels one task renders (`docs/34` section 5.7): fixed, whatever the thread count.
pub const CHUNK_PIXELS: usize = 256;

impl CandidateRenderer for ChainCandidates<'_> {
    fn sample_len(&self) -> usize {
        self.photo.sample.len()
    }

    fn render(&mut self, x: &Params, out: &mut [Px]) -> Result<(), Refused> {
        let node = self.node_for(x);
        let params = reference_node_params(&node, NodeMasks::default()).map_err(|_| Refused)?;
        self.chain
            .replace_node(self.photo.node, &params)
            .map_err(|_| Refused)?;
        let (chain, photo, conv) = (&self.chain, self.photo, &self.conv);
        let masks = &*photo.masks;
        out.par_chunks_mut(CHUNK_PIXELS)
            .enumerate()
            .for_each(|(c, chunk)| {
                for (j, d) in chunk.iter_mut().enumerate() {
                    let i = c * CHUNK_PIXELS + j;
                    let (x, y) = photo.sample[i];
                    let spot = photo.spots[i].with_raster(masks, x, y);
                    let codes = chain
                        .codes_from_node(&photo.states[i], Some(spot), photo.node)
                        .unwrap_or([0.0; 3]);
                    *d = conv.px_of_codes(codes);
                }
            });
        Ok(())
    }
}

/// The reference's pixels and scene at its analysis picture, measured: every pixel (`docs/34` section 5.1).
fn reference_side(picture: &ReferencePicture, scene: &ReferenceScene) -> (Vec<Px>, PixelScene) {
    let a = picture.analysis();
    let conv = LabConv::new();
    let px: Vec<Px> = a.linear.par_iter().map(|&l| conv.px(l)).collect();
    let mut ps = PixelScene::blank(px.len());
    for y in 0..a.height {
        for x in 0..a.width {
            let i = y * a.width + x;
            let pos = [
                (x as f64 + 0.5) / a.width as f64,
                (y as f64 + 0.5) / a.height as f64,
            ];
            let (class, skin, subject) = scene.scene.class_at(pos);
            ps.class[i] = class;
            ps.skin[i] = skin;
            ps.subject[i] = subject;
        }
    }
    (px, ps)
}

/// The network's file, read once.
fn network() -> Result<&'static Network, FitFailure> {
    static NETWORK: OnceLock<Result<Network, String>> = OnceLock::new();
    NETWORK
        .get_or_init(|| Network::shipped().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| FitFailure::Failed(e.clone()))
}

// ---------------------------------------------------------------------------------------------
// The held reference
// ---------------------------------------------------------------------------------------------

/// A reference the core holds for one node, for the session: its picture and, once a fit has made it, its scene.
pub struct HeldReference {
    picture: ReferencePicture,
    label: String,
    scene: Mutex<Option<Arc<ReferenceScene>>>,
}

impl std::fmt::Debug for HeldReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeldReference")
            .field("label", &self.label)
            .field("picture", &self.picture)
            .finish_non_exhaustive()
    }
}

impl HeldReference {
    /// The reference's scene, made on the first call and kept for the next ones.
    fn scene_for(
        &self,
        renderer: &Renderer,
        cancel: &CancelToken,
    ) -> Result<Arc<ReferenceScene>, RenderError> {
        let mut slot = self.scene.lock().unwrap_or_else(PoisonError::into_inner);
        match slot.as_ref() {
            Some(s) => Ok(Arc::clone(s)),
            None => {
                let made = Arc::new(renderer.reference_scene(&self.picture, cancel)?);
                *slot = Some(Arc::clone(&made));
                Ok(made)
            }
        }
    }
}

/// A held reference as the fit measures it: every pixel of its analysis picture and its scene (`docs/34` section 5.1).
#[derive(Debug, Clone)]
pub struct MeasuredReference {
    /// The analysis picture's width in pixels.
    pub width: usize,
    /// Its height.
    pub height: usize,
    /// Every pixel, row-major, measured.
    pub px: Vec<Px>,
    /// The reference's element, skin and subject at each pixel.
    pub scene: PixelScene,
}

/// What the card shows of a held reference: a label and the notes. The page never gets the picture itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceInfo {
    /// The name shown under the thumbnail: the file's name, or what the caller named the strip photo.
    pub label: String,
    /// Whether the reference is a file or a photo of the strip.
    pub source: ReferenceSource,
    /// The sentences of the notes ("Black bars were removed from the reference.", and so on), in a fixed order.
    pub notes: Vec<String>,
    /// Whether the picture has no colour (it is matched in black and white).
    pub grey: bool,
    /// The scene picture's size in pixels.
    pub size: (usize, usize),
}

/// The references the core holds: one per Reference Grade node of a photo, for the session.
#[derive(Debug, Default)]
pub struct ReferenceHold {
    held: Mutex<BTreeMap<(PhotoId, u32), Arc<HeldReference>>>,
}

impl ReferenceHold {
    fn map(&self) -> std::sync::MutexGuard<'_, BTreeMap<(PhotoId, u32), Arc<HeldReference>>> {
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn get(&self, photo: PhotoId, node: u32) -> Option<Arc<HeldReference>> {
        self.map().get(&(photo, node)).cloned()
    }

    fn put(&self, photo: PhotoId, node: u32, r: HeldReference) -> ReferenceInfo {
        let info = info_of(&r);
        self.map().insert((photo, node), Arc::new(r));
        info
    }

    /// Lets go of a node's reference. Returns whether there was one.
    pub fn clear(&self, photo: PhotoId, node: u32) -> bool {
        self.map().remove(&(photo, node)).is_some()
    }

    /// Lets go of every reference of a photo (it left the session).
    pub fn clear_photo(&self, photo: PhotoId) {
        self.map().retain(|&(p, _), _| p != photo);
    }

    /// Lets go of every reference.
    pub fn clear_all(&self) {
        self.map().clear();
    }

    /// Lets go of the references of the photo's nodes that are no longer Reference Grade nodes of `settings`.
    pub fn prune(&self, photo: PhotoId, settings: &DevelopSettings) {
        let mut held = self.map();
        if held.is_empty() {
            return;
        }
        held.retain(|&(p, node), _| {
            p != photo
                || settings.nodes.kind_of(NodeId::Corrector(node)) == Some(NodeKind::ReferenceGrade)
        });
    }

    /// How many references are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map().len()
    }

    /// Whether none is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map().is_empty()
    }
}

fn info_of(r: &HeldReference) -> ReferenceInfo {
    ReferenceInfo {
        label: r.label.clone(),
        source: r.picture.source(),
        notes: r
            .picture
            .notes()
            .iter()
            .map(|n: &ReferenceNote| n.text().to_owned())
            .collect(),
        grey: r.picture.is_grey(),
        size: (r.picture.scene().width, r.picture.scene().height),
    }
}

// ---------------------------------------------------------------------------------------------
// The fit's events, outcome and failures
// ---------------------------------------------------------------------------------------------

/// Where a fit is, as the card says it (`docs/34` section 9.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FitEvent {
    /// Making the scene of the photo or of the reference: "Finding what is in the pictures…".
    Scene,
    /// Searching: the search's number (1 to 3) and how far the fit is, by stage (an estimate that never goes back).
    Search {
        /// The search, 1 to 3.
        search: u8,
        /// The number of searches (3).
        of: u8,
        /// How far the fit is, 0 to 1.
        fraction: f64,
        /// The candidates evaluated so far.
        evals: usize,
    },
}

impl FitEvent {
    /// The card's line for the event.
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            FitEvent::Scene => "Finding what is in the pictures…".to_owned(),
            FitEvent::Search { search, of, .. } => format!("Fitting… {search} of {of}"),
        }
    }
}

/// How a fit ended when it wrote something.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitStatus {
    /// The node holds a fit: "Fitted to the reference."
    Fitted,
    /// The result was no better than the photo as it is, so the node was given the identity: "The photo is already close to the
    /// reference. Nothing was changed."
    AlreadyClose,
}

impl FitStatus {
    /// The card's status line.
    #[must_use]
    pub fn text(self) -> &'static str {
        match self {
            FitStatus::Fitted => "Fitted to the reference.",
            FitStatus::AlreadyClose => {
                "The photo is already close to the reference. Nothing was changed."
            }
        }
    }
}

/// One element as the fit found it in the two pictures (`docs/34` section 4b): its share of each picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementFound {
    /// The element's name: `sky`, `vegetation`, `water`, `ground`, `built` or `skin`.
    pub element: &'static str,
    /// Its share of the photo's analysis picture, 0 to 1.
    pub photo_share: f64,
    /// Its share of the reference's, 0 to 1.
    pub reference_share: f64,
}

/// What a fit did.
#[derive(Debug, Clone)]
pub struct FitOutcome {
    /// How it ended.
    pub status: FitStatus,
    /// The notes for the card: "No skin was found, so Skin protection did nothing." and the reference's own.
    pub notes: Vec<String>,
    /// The patch that was applied to the photo's settings: the node's fitted fields, in one patch.
    pub applied: SettingsPatch,
    /// The patch that takes them back (one undo step).
    pub undo: SettingsPatch,
    /// The fit's numbers, its objective and the three searches (for reports and tests).
    pub result: Box<FitResult>,
    /// The six elements and their shares of the two pictures (`ym render --json`, the evaluation tool).
    pub found: Vec<ElementFound>,
    /// Whether skin was found in the photo (its share is at least 2 %): when not, Skin protection did nothing.
    pub skin_found: bool,
}

/// Why a fit wrote nothing. The `Display` text is the sentence of `docs/34` section 9.3.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FitFailure {
    /// No reference was chosen for the node.
    #[error("Choose a reference first.")]
    NoReference,
    /// The node is not a Reference Grade node of the photo.
    #[error("Node {0} is not a Reference Grade node.")]
    NotReferenceGrade(u32),
    /// The node does not take part in the photo's picture.
    #[error(
        "This node isn't connected to the Output, so it doesn't change the picture. Connect it, then fit."
    )]
    NotConnected,
    /// The reference cannot be used.
    #[error("The reference can't be used: {0}")]
    Reference(ReferenceError),
    /// This install has no scene model.
    #[error("The scene model is missing from this install. Reinstall Yellowmoon.")]
    SceneModelMissing,
    /// The person pressed Cancel: nothing was changed.
    #[error("The fit was cancelled. Nothing was changed.")]
    Cancelled,
    /// The grade could not be fitted (the detail is for the log).
    #[error("The grade couldn't be fitted. Try another reference.")]
    Failed(String),
    /// The photo is not open.
    #[error("{0}")]
    Photo(String),
}

impl From<RenderError> for FitFailure {
    fn from(e: RenderError) -> Self {
        match e {
            RenderError::Cancelled => FitFailure::Cancelled,
            RenderError::Mask(s) if s == NO_SCENE_MODEL_SENTENCE || s == NO_SKIN_MODEL_SENTENCE => {
                FitFailure::SceneModelMissing
            }
            RenderError::Color(s) if s.contains("not connected to the Output") => {
                FitFailure::NotConnected
            }
            other => FitFailure::Failed(other.to_string()),
        }
    }
}

impl From<SnapshotError> for FitFailure {
    fn from(e: SnapshotError) -> Self {
        match e {
            SnapshotError::Reference(r) => FitFailure::Reference(r),
            SnapshotError::Render(r) => r.into(),
        }
    }
}

impl From<SessionError> for FitFailure {
    fn from(e: SessionError) -> Self {
        FitFailure::Photo(e.to_string())
    }
}

impl From<FitError> for FitFailure {
    fn from(e: FitError) -> Self {
        match e {
            FitError::Cancelled => FitFailure::Cancelled,
            other => FitFailure::Failed(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------------------------

/// Fits the Reference Grade node `node` of `src` to the held reference, without writing anything: the result's numbers and what the
/// patch would be. `cancel` is asked before every candidate; `progress` is told where the fit is.
///
/// # Errors
///
/// [`FitFailure`]: `Cancelled` when the token says so, `SceneModelMissing`, `NotConnected`, and `Failed` for everything else.
pub fn fit_node(
    renderer: &Renderer,
    src: &PhotoSource,
    node: u32,
    held: &HeldReference,
    cancel: &CancelToken,
    progress: &mut dyn FnMut(FitEvent),
) -> Result<(FitResult, Vec<String>, Vec<ElementFound>), FitFailure> {
    let Some(base) = src.settings.nodes.reference_grade(node) else {
        return Err(FitFailure::NotReferenceGrade(node));
    };
    let net = network()?;
    progress(FitEvent::Scene);
    let photo = renderer.reference_fit_photo(src, node, cancel)?;
    let reference_scene = held.scene_for(renderer, cancel)?;
    if cancel.is_cancelled() {
        return Err(FitFailure::Cancelled);
    }
    let (r_px, r_scene) = reference_side(&held.picture, &reference_scene);
    let mut candidates = ChainCandidates::new(&photo, base);
    // The photo as it stands: the sample with the node at its identity.
    let mut p_px = vec![Px::default(); candidates.sample_len()];
    candidates
        .render(&[0.0; ym_color::refgrade::N_PARAMS], &mut p_px)
        .map_err(|_| {
            FitFailure::Failed("The photo could not be rendered for the fit.".to_owned())
        })?;
    let fixed = Fixed::new(
        net,
        &p_px,
        &photo.scene,
        &r_px,
        &r_scene,
        base.skin_protection,
    )
    .map_err(|e| FitFailure::Failed(e.to_string()))?;
    let setup = FitSetup {
        fixed: &fixed,
        scene: &photo.scene,
        mask_ok: photo.mask_ok,
    };
    let found: Vec<ElementFound> = ["sky", "vegetation", "water", "ground", "built", "skin"]
        .into_iter()
        .zip(fixed.p.el.iter().zip(&fixed.r.el))
        .map(|(element, (p, r))| ElementFound {
            element,
            photo_share: p.share,
            reference_share: r.share,
        })
        .collect();
    let mut notes: Vec<String> = Vec::new();
    // Spatial nodes after this one (Film, Blur, Face Refinement) make their picture before the colour pass: the fit does not run them.
    {
        let params = photo.chain.params();
        let mut after = false;
        let mut spatial = false;
        for step in params.nodes.plan.steps() {
            if step.node == node {
                after = true;
            } else if after && params.films.contains_key(&step.node) {
                spatial = true;
            }
        }
        if spatial {
            notes.push(
                "Nodes after this one that blur or add film aren't part of the fit.".to_owned(),
            );
        }
    }
    if base.skin_protection && fixed.p.el[5].share < ym_color::refgrade::measure::PRESENT_SHARE {
        notes.push("No skin was found, so Skin protection did nothing.".to_owned());
    }
    let result = fit(
        &setup,
        &mut candidates,
        &|| cancel.is_cancelled(),
        &mut |p: FitProgress| {
            progress(FitEvent::Search {
                search: p.search,
                of: p.searches,
                fraction: p.fraction(),
                evals: p.evals,
            });
        },
    )?;
    Ok((result, notes, found))
}

/// The settings with the node's fitted fields set to a fit's numbers.
fn with_numbers(settings: &DevelopSettings, node: u32, n: &Numbers) -> Option<DevelopSettings> {
    let mut next = settings.clone();
    apply_numbers(next.nodes.reference_grade_mut(node)?, n);
    Some(next)
}

impl Core {
    /// Chooses an image file (JPEG, PNG or TIFF) as the reference of a Reference Grade node. The file is only read; the core holds
    /// the reference for the session and writes nothing of it anywhere. `label` is what the card shows under the thumbnail (the
    /// file's name when `None`).
    ///
    /// # Errors
    ///
    /// [`FitFailure::Reference`] with the sentence of `docs/34` section 9.3 for a file that cannot be used,
    /// [`FitFailure::NotReferenceGrade`] for a node that is not one.
    pub fn reference_choose_file(
        &self,
        photo: PhotoId,
        node: u32,
        path: &Path,
        label: Option<&str>,
    ) -> Result<ReferenceInfo, FitFailure> {
        self.check_reference_node(photo, node)?;
        let picture = ReferencePicture::from_file(path).map_err(FitFailure::Reference)?;
        let label = label.map(str::to_owned).unwrap_or_else(|| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
        Ok(self.references.put(
            photo,
            node,
            HeldReference {
                picture,
                label,
                scene: Mutex::new(None),
            },
        ))
    }

    /// Chooses another photo of the session as the reference: it is rendered through its own settings and look (a snapshot, so
    /// later edits to it change nothing). The photo being graded refuses itself.
    ///
    /// # Errors
    ///
    /// [`FitFailure::Reference`] ("That's the photo you're grading. Choose another."), [`FitFailure::Photo`] for a photo that is not
    /// open or not ready, [`FitFailure::Cancelled`], [`FitFailure::NotReferenceGrade`].
    pub fn reference_choose_photo(
        &self,
        photo: PhotoId,
        node: u32,
        other: PhotoId,
        label: &str,
        cancel: &CancelToken,
    ) -> Result<ReferenceInfo, FitFailure> {
        self.check_reference_node(photo, node)?;
        let src = PhotoSource::from_session(&self.session, other)?;
        let picture = self.renderer.reference_snapshot(photo, &src, cancel)?;
        Ok(self.references.put(
            photo,
            node,
            HeldReference {
                picture,
                label: label.to_owned(),
                scene: Mutex::new(None),
            },
        ))
    }

    /// The photo as the evaluation tool measures it: `Renderer::reference_eval_photo` for the node of an open photo (every pixel of
    /// the analysis picture, not the fit's sample). Made once per photo; `docs/34` section 10.1.
    ///
    /// # Errors
    ///
    /// [`FitFailure`], as for [`Core::reference_fit`].
    pub fn reference_eval_photo(
        &self,
        photo: PhotoId,
        node: u32,
        cancel: &CancelToken,
    ) -> Result<FitPhoto, FitFailure> {
        let src = PhotoSource::from_session(&self.session, photo)?;
        Ok(self.renderer.reference_eval_photo(&src, node, cancel)?)
    }

    /// The node's held reference as the fit measures it (its scene is made now when no fit has made it): the evaluation tool's
    /// reference side. Nothing is written.
    ///
    /// # Errors
    ///
    /// [`FitFailure::NoReference`] when none is held, and the errors of the scene.
    pub fn reference_measured(
        &self,
        photo: PhotoId,
        node: u32,
        cancel: &CancelToken,
    ) -> Result<MeasuredReference, FitFailure> {
        let held = self
            .references
            .get(photo, node)
            .ok_or(FitFailure::NoReference)?;
        let scene = held.scene_for(&self.renderer, cancel)?;
        let (px, ps) = reference_side(&held.picture, &scene);
        let a = held.picture.analysis();
        Ok(MeasuredReference {
            width: a.width,
            height: a.height,
            px,
            scene: ps,
        })
    }

    /// What the card shows of the node's held reference, when it has one.
    #[must_use]
    pub fn reference_info(&self, photo: PhotoId, node: u32) -> Option<ReferenceInfo> {
        self.references.get(photo, node).map(|h| info_of(&h))
    }

    /// The node's held reference as a small picture for its card: the scene picture (black bars already removed) reduced by a box
    /// average so that its long edge is at most `long_edge_px`, as sRGB display codes, row by row: `(width, height, rgb8)`. `None`
    /// when the node holds no reference. The picture is a view of what is held, made on demand; nothing is kept or written.
    #[must_use]
    pub fn reference_thumbnail(
        &self,
        photo: PhotoId,
        node: u32,
        long_edge_px: usize,
    ) -> Option<(usize, usize, Vec<u8>)> {
        let held = self.references.get(photo, node)?;
        let scene = held.picture.scene();
        let (w, h) = (scene.width, scene.height);
        if w == 0 || h == 0 || long_edge_px == 0 {
            return None;
        }
        let codes = scene.srgb8();
        let scale = (long_edge_px as f64 / w.max(h) as f64).min(1.0);
        let (tw, th) = (
            ((w as f64 * scale).round() as usize).max(1),
            ((h as f64 * scale).round() as usize).max(1),
        );
        let mut out = Vec::with_capacity(tw * th * 3);
        for ty in 0..th {
            let (y0, y1) = (
                ty * h / th,
                ((ty + 1) * h).div_ceil(th).min(h).max(ty * h / th + 1),
            );
            for tx in 0..tw {
                let (x0, x1) = (
                    tx * w / tw,
                    ((tx + 1) * w).div_ceil(tw).min(w).max(tx * w / tw + 1),
                );
                let mut sum = [0u32; 3];
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = (y * w + x) * 3;
                        for c in 0..3 {
                            sum[c] += u32::from(codes[i + c]);
                        }
                    }
                }
                let n = ((y1 - y0) * (x1 - x0)) as u32;
                for c in sum {
                    out.push(((c + n / 2) / n) as u8);
                }
            }
        }
        Some((tw, th, out))
    }

    /// Lets go of the node's reference (`clear_reference`). Returns whether there was one.
    pub fn reference_clear(&self, photo: PhotoId, node: u32) -> bool {
        self.references.clear(photo, node)
    }

    fn check_reference_node(&self, photo: PhotoId, node: u32) -> Result<(), FitFailure> {
        let view = self.session.view(photo)?;
        if view.settings.nodes.kind_of(NodeId::Corrector(node)) != Some(NodeKind::ReferenceGrade) {
            return Err(FitFailure::NotReferenceGrade(node));
        }
        Ok(())
    }

    /// Fits a Reference Grade node to its reference and writes the result as one patch (`fit_reference`). Blocks the calling thread
    /// for the length of the fit (several seconds); `progress` is told where it is, and `cancel` stops it between candidates: a
    /// cancel, like a failure, writes nothing. The fit starts again from the identity for the fitted fields and leaves the node's other
    /// fields alone; a result no better than the photo as it is writes the identity and says so ([`FitStatus::AlreadyClose`]).
    ///
    /// # Errors
    ///
    /// [`FitFailure`] with the sentence of `docs/34` section 9.3.
    pub fn reference_fit(
        &self,
        photo: PhotoId,
        node: u32,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(FitEvent),
    ) -> Result<FitOutcome, FitFailure> {
        let held = self
            .references
            .get(photo, node)
            .ok_or(FitFailure::NoReference)?;
        let src = PhotoSource::from_session(&self.session, photo)?;
        let (result, mut notes, found) =
            fit_node(&self.renderer, &src, node, &held, cancel, progress)?;
        if cancel.is_cancelled() {
            return Err(FitFailure::Cancelled);
        }
        let next = with_numbers(&src.settings, node, &result.numbers)
            .ok_or(FitFailure::NotReferenceGrade(node))?;
        // One patch of the fitted fields, against what the photo holds now (a hand edit made while the fit ran stays).
        let now = self.session.view(photo)?.settings;
        let patch_settings = with_numbers(&now, node, &result.numbers).unwrap_or(next);
        let applied = SettingsPatch::between(&now, &patch_settings);
        let undo = SettingsPatch::between(&patch_settings, &now);
        self.update_settings(photo, &applied)?;
        notes.extend(info_of(&held).notes);
        let status = if result.identity {
            FitStatus::AlreadyClose
        } else {
            FitStatus::Fitted
        };
        let skin_found = found
            .last()
            .is_some_and(|e| e.photo_share >= ym_color::refgrade::measure::PRESENT_SHARE);
        Ok(FitOutcome {
            status,
            notes,
            applied,
            undo,
            result: Box::new(result),
            found,
            skin_found,
        })
    }
}
